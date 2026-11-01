# ECE198 Motion Tracker

A real-time exercise motion tracking system developed for
ECE198 at the University of Waterloo.

## Project Overview

The system uses a USB camera and Google's MediaPipe
Pose Landmarker to detect human body landmarks and
calculate knee joint angles in real time.

An Arduino UNO R4 will process the angle data and
provide feedback to the user.

## Current Progress

- [x] USB camera integration
- [x] MediaPipe Full pose detection
- [x] Left and right knee angle calculation
- [x] Basic repetition counting prototype
- [ ] Improved detection stability
- [ ] Python-to-Arduino serial communication
- [ ] Arduino feedback system

## Planned Serial Protocol

Format: K1,left_angle,right_angle

Example: K1,92.4,175.6

- Left and right angles are measured in degrees.
- -1 indicates unreliable or unavailable data.
- Planned baud rate: 115200
- Planned update rate: 20 Hz

## Setup

Python 3.14 is used for development.

Install dependencies:

    pip install -r requirements.txt

Download the MediaPipe Full Pose Landmarker model
from Google's official MediaPipe model repository
and save it as pose_landmarker.task.

Run the pose detection prototype:

    python pose_test.py

Run the knee angle tracker:

    python angle_test.py

## Project Status

Work in progress.
