#!/usr/bin/env python3
"""realtime_slip_detector_v3.py - live slip prevention with the trained model (all berry sizes).
terminal 1: ~/launch_leap.sh
terminal 2: source ~/xense_env/bin/activate; source ~/leap_ws/devel/setup.bash
            python3 ~/grasp_dataset/standalone/realtime_slip_detector_v3.py [--test-loosen]"""
import os, sys, time, csv, threading, argparse
import numpy as np


def _enable_gpu():
    import ctypes, glob
    root = os.path.expanduser(os.environ.get("XENSE_CUDA_LIBS", "~/cuda11_libs"))
    libs = sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib", "lib*.so*")))
    if not libs:
        print(f"[gpu] no CUDA libraries found under {root} - running on CPU"); return
    pending, loaded = list(libs), 0
    for _ in range(6):
        still = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL); loaded += 1
            except OSError:
                still.append(path)
        if len(still) == len(pending):
            break
        pending = still
    print(f"[gpu] loaded {loaded}/{len(libs)} CUDA libraries from {root}")
    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"[gpu] onnxruntime import failed: {e}"); return
    orig = ort.InferenceSession

    class CudaSession(orig):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
            super().__init__(path_or_bytes, sess_options,
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"], **kw)
            print(f"[gpu] onnxruntime session providers: {self.get_providers()}")

    ort.InferenceSession = CudaSession


if "--cpu" not in sys.argv:
    _enable_gpu()

import rospy
from sensor_msgs.msg import JointState
import leap_hand_utils.leap_hand_utils as lhu
from xensesdk import Sensor

sys.path.insert(0, os.path.expanduser("~/grasp_dataset/standalone"))
import slip_features as SF

INDEX_SENSOR_ID, THUMB_SENSOR_ID = "OG000961", "OG000324"
MODEL = os.path.expanduser("~/grasp_dataset/slip_model_v4_np.npz")
WEIGHTS = {1: 0.5, 2: 0.5, 3: 1.0, 14: 0.5, 15: 1.0}


def make_msg(pose):
    m = JointState()
    m.header.stamp = rospy.Time.now()
    m.name = [f"joint_{i}" for i in range(16)]
    m.position = [float(v) for v in pose]
    return m


class XenseReader(threading.Thread):
    def __init__(self, sensors, t0):
        super().__init__(daemon=True)
        self.sensors, self.t0 = sensors, t0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.fz = [0.0, 0.0]
        self.n = 0
        self.reset(time.time())

    def reset(self, t_start):
        with self.lock:
            self.t_start = t_start - self.t0
            self.refs = [SF.ContactRef(), SF.ContactRef()]
            self.fs = SF.FeatureStream(self.t_start)
            self.ref_failed = [None, None]

    def run(self):
        while not self.stop.is_set() and not rospy.is_shutdown():
            for k, s in enumerate(self.sensors):
                try:
                    F, M = s.selectSensorInfo(Sensor.OutputType.Force, Sensor.OutputType.Marker2D)
                except Exception as e:
                    rospy.logwarn_throttle(2, f"[Xense] read failed: {e}")
                    continue
                if F is None or M is None:
                    continue
                F = np.asarray(F, float)
                M = np.asarray(M, float)[:, :, :2]
                t = time.time() - self.t0
                with self.lock:
                    self.fz[k] = float(np.abs(F[..., 2]).sum())
                    r = self.refs[k]
                    if not r.ok:
                        if self.ref_failed[k]:
                            continue
                        r.add(F, M)
                        if t - self.t_start >= SF.REF_WIN:
                            if not r.finish():
                                self.ref_failed[k] = "contact patch too small"
                                rospy.logwarn(f"[ref] {('index', 'thumb')[k]}: contact patch too small - no contact?")
                        continue
                    self.fs.push(t, ("index", "thumb")[k], r.shear(F, M))
                    self.n += 1

    def features(self):
        with self.lock:
            t = time.time() - self.t0
            return t, self.fs.vector(t), list(self.fz)


class SlipPrevention:
    def __init__(self, a):
        self.a = a
        rospy.init_node("realtime_slip_detector_v5")
        self.G = SF.NumpyGBM(MODEL)
        self.thr = a.threshold if a.threshold else self.G.threshold
        rospy.loginfo(f"[Model] {MODEL}: threshold {self.thr:.3f}, {a.consecutive} consecutive at {a.rate:.0f} Hz")
        self.pub = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=1)
        self.sim_min, self.sim_max = lhu.LEAP_limits()
        rospy.loginfo("[Xense] initialising sensors ...")
        si = Sensor.create(INDEX_SENSOR_ID); rospy.sleep(2.0)
        st = Sensor.create(THUMB_SENSOR_ID); rospy.sleep(2.0)
        self.sensors = [si, st]
        self.t0 = time.time()
        os.makedirs(os.path.expanduser("~/grasp_data_live"), exist_ok=True)
        fn = os.path.expanduser(f"~/grasp_data_live/live_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.fh = open(fn, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["t_s", "p_slip", "threshold", "consecutive", "fz_index", "fz_thumb", "loosen_rad", "event"])
        rospy.loginfo(f"[Log] {fn}")

    def publish(self, pose, n=1, hz=30):
        r = rospy.Rate(hz)
        for _ in range(n):
            self.pub.publish(make_msg(np.clip(pose, self.sim_min, self.sim_max))); r.sleep()

    def read_fz(self):
        out = []
        for sen in self.sensors:
            try:
                F = np.asarray(sen.selectSensorInfo(Sensor.OutputType.Force), float)
                out.append(float(np.abs(F[..., 2]).sum()))
            except Exception:
                out.append(0.0)
        return out

    def grasp(self):
        open_pose = np.zeros(16)
        # v5: same pinch pose as data collection; + on joints 3/15 closes the pinch
        self.grip_pose = np.array([0.057, 1.800, 0.059, -0.268,  -0.028, -0.005, -0.004, -0.000,
                                   -0.028, 0.047, -0.086, 0.041,  1.533, 0.041, 0.434, -0.426])
        rospy.loginfo("[Setup] opening hand ...")
        self.publish(open_pose, 60)
        rospy.loginfo("[Setup] moving to pinch pose - PLACE THE BERRY between the fingertips and keep holding it ...")
        r = rospy.Rate(30)
        for i in range(81):
            self.pub.publish(make_msg(np.clip(open_pose + i / 80 * (self.grip_pose - open_pose), self.sim_min, self.sim_max)))
            r.sleep()
        self.pose = self.grip_pose.copy()
        self.publish(self.pose, 90)
        rospy.loginfo(f"[Setup] squeezing to {self.a.target_force:.1f} N (as in data collection) ...")
        for k in range(10):
            self.pose[3] += 0.05; self.pose[15] += 0.05
            self.publish(self.pose, 8)
            fz = self.read_fz()
            rospy.loginfo(f"    squeeze {k}: +{0.05 * (k + 1):.2f} rad  Fz index={fz[0]:.2f} thumb={fz[1]:.2f} N")
            if min(fz) >= self.a.target_force or max(fz) >= self.a.max_force:
                break
        input("[Setup] berry held? REMOVE YOUR HAND, then press ENTER ...")
        self.publish(self.pose, 60)

    def tighten(self, fz):
        if max(fz) > self.a.max_force:
            rospy.logwarn(f"[Safety] force {max(fz):.2f} N > {self.a.max_force} N - not tightening")
            return False
        for j, w in WEIGHTS.items():
            self.pose[j] += w * self.a.step
        self.publish(self.pose, 5)
        rospy.logwarn(f"[Reaction] tightened {self.a.step:+.3f} rad (weighted joints {list(WEIGHTS)})")
        return True

    def run(self):
        self.grasp()
        rd = XenseReader(self.sensors, self.t0)
        rd.start()
        rospy.loginfo(f"[Monitor] reference {SF.REF_WIN} s + baseline {SF.BASE} s - keep everything still ...")
        rate = rospy.Rate(self.a.rate)
        consec, loosen, loosening, armed_msg = 0, 0.0, self.a.test_loosen, False
        last_log, n_react = 0.0, 0
        try:
            while not rospy.is_shutdown():
                t, x, fz = rd.features()
                ev = ""
                if x is None:
                    self.pub.publish(make_msg(np.clip(self.pose, self.sim_min, self.sim_max)))
                    rate.sleep(); continue
                if not armed_msg:
                    rospy.loginfo(f"[Monitor] slip detection ACTIVE (sensor frames so far {rd.n})"
                                  + ("  - test loosening ON" if loosening else ""))
                    armed_msg = True
                p = self.G.predict_p(x)
                consec = consec + 1 if p > self.thr else 0
                if 0 < consec < self.a.consecutive:
                    rospy.loginfo(f"[Slip detected] {consec}/{self.a.consecutive} consecutive (p={p:.3f}) - waiting for confirmation")
                if consec >= self.a.consecutive:
                    rospy.logwarn(f"[SLIP CONFIRMED] {consec} consecutive, p={p:.3f}, Fz index={fz[0]:.2f} thumb={fz[1]:.2f} N")
                    if loosening:
                        loosening = False
                        rospy.logwarn(f"[Test] loosening stopped at {loosen:.3f} rad")
                    ev = "slip"
                    if self.tighten(fz):
                        n_react += 1
                        ev = "slip+tighten"
                        rospy.sleep(self.a.settle)
                        rd.reset(time.time())
                        armed_msg = False
                    consec = 0
                elif loosening:
                    d = self.a.loosen_rate / self.a.rate
                    for j, w in WEIGHTS.items():
                        self.pose[j] -= w * d
                    loosen += d
                self.pub.publish(make_msg(np.clip(self.pose, self.sim_min, self.sim_max)))
                self.w.writerow([round(t, 3), round(p, 4), round(self.thr, 4), consec, round(fz[0], 3), round(fz[1], 3),
                                 round(loosen, 4), ev])
                if t - last_log >= 2.0:
                    self.fh.flush(); last_log = t
                    rospy.loginfo(f"[Status] t={t:6.1f}s p={p:.3f} consec={consec} Fz i={fz[0]:.2f} t={fz[1]:.2f} N"
                                  + (f" loosen={loosen:.3f} rad" if self.a.test_loosen else "") + f" reactions={n_react}")
                rate.sleep()
        finally:
            rd.stop.set()
            self.fh.close()
            rospy.loginfo(f"[Done] reactions: {n_react}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-loosen", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--loosen-rate", type=float, default=0.01)
    ap.add_argument("--grip", type=float, default=-0.25)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--target-force", type=float, default=1.5, help="N per finger after the squeeze")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-force", type=float, default=3.5)
    ap.add_argument("--settle", type=float, default=0.5)
    a = ap.parse_args(rospy.myargv()[1:])
    try:
        SlipPrevention(a).run()
    except rospy.ROSInterruptException:
        pass
