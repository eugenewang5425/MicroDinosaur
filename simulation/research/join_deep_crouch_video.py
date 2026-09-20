"""Combine the two independently verified, preselected seed1 replay clips."""
import json
import imageio.v2 as imageio
import numpy as np
from evaluate_deep_crouch import OUT


def main():
    base = OUT/'videos'
    for label in ('shallow', 'deep'):
        assert json.loads((base/label/'verification.json').read_text())['metrics_match']
    a = imageio.get_reader(str(base/'shallow/simulation.mp4'))
    b = imageio.get_reader(str(base/'deep/simulation.mp4'))
    writer = imageio.get_writer(str(base/'comparison.mp4'), fps=25, codec='libx264', quality=8)
    frames = 0
    try:
        for left, right in zip(a, b, strict=True):
            writer.append_data(np.concatenate((left, right), axis=1)); frames += 1
    finally: writer.close(); a.close(); b.close()
    assert frames == 300
    print(base/'comparison.mp4')


if __name__ == '__main__': main()
