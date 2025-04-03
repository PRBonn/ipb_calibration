FROM ubuntu:20.04
#FROM python:3.10.12 Note: Unfortunately with this the 3D visualization with open3d does not work.

LABEL maintainer="Thomas Laebe <laebe@ipb.uni-bonn.de>"

# -----------------------------------------------------------------------------------------
# Needed Packages

# Python
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip # Note that without this the installation
                                                        # of the source package with "-e" does not work

# Libraries needed for open3d
RUN apt-get update && apt-get install --no-install-recommends -y libc++1 libgomp1  libpng16-16 libglfw3

# For building apriltag library 
RUN apt-get update && apt-get install -y git
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake

# For opencv (cv2)
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# -----------------------------------------------------------------------------------------
# We need the apriltag library. Note that this only works AFTER the source package install. Unclear why
# The version you get with "pip install apriltag" is too old
# (it has some different point numbering, thus the calibration may fail!). We use the github
# repo. Note that we have to do this AFTER the installation of the calibration package,
# because the build system checks for numpy which is not there before installing the calibration package.
WORKDIR /root
RUN git clone --branch 'v3.4.2' https://github.com/AprilRobotics/apriltag

WORKDIR /root/apriltag
# numpy needs to be there otherwise python wrapper is not built
RUN pip install numpy
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target install

# Library path has to be set so that python finds the apriltag library
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages

# HACK: The build does not name the python wrapper library correctly. Correct this here.
RUN mv /usr/local/lib/python3.8/site-packages/apriltag..so /usr/local/lib/python3.8/site-packages/apriltag.cpython-38-x86_64-linux-gnu.so


# -----------------------------------------------------------------------------------------
# The actual program

# Python debugger which is actually not needed but may appear in source: ipdb
RUN pip --no-cache-dir install ipdb

# create folder for code and go to it
WORKDIR /root/calib

# Copy the neccessary files/dirs into the container
COPY scripts ./scripts
COPY src ./src
COPY pyproject.toml .

# Install the calibration python package and it's (python) dependencies
WORKDIR /root/calib
RUN pip --no-cache-dir install -e .



# =============================================================================
# Now do the actual work: Start the camera system and LiDAR calibration
WORKDIR /root
COPY entrypoint_apriltag_extraction.bash /root/entrypoint_apriltag_extraction.bash

CMD /root/entrypoint_apriltag_extraction.bash

