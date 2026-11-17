import os
src = os.path.expanduser("~/grasp_dataset/standalone/realtime_slip_detector_v4.py")
dst = os.path.expanduser("~/grasp_dataset/standalone/realtime_slip_detector_v5.py")
s = open(src).read()
old_grasp = s[s.index("    def grasp(self):"):s.index("    def tighten(self, fz):")]
new_grasp = '''    def read_fz(self):
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

'''
s = s.replace(old_grasp, new_grasp)
old_c = "                consec = consec + 1 if p > self.thr else 0\n"
s = s.replace(old_c, old_c + "                if 0 < consec < self.a.consecutive:\n"
              "                    rospy.loginfo(f\"[Slip detected] {consec}/{self.a.consecutive} consecutive (p={p:.3f}) - waiting for confirmation\")\n")
old_a = '    ap.add_argument("--consecutive", type=int, default=3)\n'
s = s.replace(old_a, old_a + '    ap.add_argument("--target-force", type=float, default=1.5, help="N per finger after the squeeze")\n')
s = s.replace('rospy.init_node("realtime_slip_detector_v3")', 'rospy.init_node("realtime_slip_detector_v5")')
open(dst, "w").write(s); os.chmod(dst, 0o755); print("written", dst)
