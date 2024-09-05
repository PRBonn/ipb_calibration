# Joint Intrinsic and Extrinsic Calibration of Perception Systems Utilizing a Calibration Environment


![image](https://github.com/user-attachments/assets/1f3d64c6-fd31-4bf7-adef-cfa7cda43e56)

<div style="width:500px">
Basically all multi-sensor systems must calibrate
their sensors to exploit their full potential for state estimation
such as mapping and localization. In this paper, we investigate
the problem of extrinsic and intrinsic calibration of perception
systems. Traditionally, targets in the form of checkerboards or
uniquely identifiable tags are used to calibrate those systems.
We propose to use a whole calibration environment as a target
that supports the intrinsic and extrinsic calibration of different
types of sensors. By doing so, we are able to calibrate multiple
perception systems with different configurations, sensor types,
and sensor modalities. Our approach does not rely on overlaps
between sensors which is often otherwise required when using
classical targets. The main idea is to relate the measurements for
each sensor to a precise model of the calibration environment.
For this, we can choose for each sensor a specific method that best
suits its calibration. Then, we estimate all intrinsics and extrinsics
jointly using least squares adjustment. For the final evaluation
of a LiDAR-to-camera calibration of our system, we propose an
evaluation method that is independent of the calibration. This
allows for quantitative evaluation between different calibration
methods. The experiments show that our proposed method is
able to provide reliable calibration.
</div>


> [!NOTE]
> We will make the code available soon after submission deadline of ICRA 2025.