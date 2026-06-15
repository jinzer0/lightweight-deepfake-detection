from __future__ import annotations

import argparse
from src.demo.detector_service import DetectorService


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Launch the Gradio GenImage detector demo.')
    p.add_argument('--config', default='configs/fusion.yaml')
    p.add_argument('--server_name', default='127.0.0.1')
    p.add_argument('--server_port', type=int, default=7860)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    import gradio as gr
    service=DetectorService(args.config)
    demo=gr.Interface(fn=service.predict, inputs=gr.Image(type='pil'), outputs='json', title='AI-GEN Image Detector')
    demo.launch(server_name=args.server_name, server_port=args.server_port)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
