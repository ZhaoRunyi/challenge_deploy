"""Read actual 640x480/30fps color profiles; never connects to a robot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pyrealsense2 as rs


SERIALS = {'cam_high':'323422071854', 'cam_left_wrist':'344322073012', 'cam_right_wrist':'335522070790'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; use a new file')
    records={}
    for name, serial in SERIALS.items():
        pipeline=rs.pipeline()
        config=rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        started=False
        try:
            profile=pipeline.start(config)
            started=True
            pipeline.wait_for_frames(timeout_ms=5000)
            stream=profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr=stream.get_intrinsics()
            device=profile.get_device()
            if device.get_info(rs.camera_info.serial_number) != serial:
                raise ValueError('Camera identity mismatch')
            records[name]=dict(serial=serial,device_name=device.get_info(rs.camera_info.name),
                firmware=device.get_info(rs.camera_info.firmware_version),
                width=intr.width,height=intr.height,fps=stream.fps(),stream='color',format='bgr8',
                camera_matrix=[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],
                distortion=list(intr.coeffs),distortion_model=str(intr.model))
        finally:
            if started:
                pipeline.stop()
    result=dict(read_at_utc=datetime.now(timezone.utc).isoformat(),source='live RealSense SDK active profiles',cameras=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
