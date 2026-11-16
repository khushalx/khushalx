# Crimson Cosmos artwork

From the repository root, install the build dependencies and regenerate the SVGs:

```sh
python3 -m pip install -r tools/requirements.txt
python3 tools/render_cosmos.py
```

The generator ray traces a thin accretion disk in Schwarzschild spacetime, then adds procedural emission, dust, stars, and optical scatter. It embeds a cropped JPEG exposure in each SVG and outlines the Space Grotesk lettering. Every asset can be rebuilt offline once the dependencies are installed. The two SVGs share the same exposure but use different crops and type layouts.

The emission colors use a ramp sampled from the profile avatar. SVG animations move soft light along the disk, carry dust and sparks, and twinkle a few stars. The README explicitly loads the `#still` variant for reduced motion, and the SVG also checks the browser preference for direct views.

Space Grotesk is vendored from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) under the [SIL Open Font License](fonts/OFL-SpaceGrotesk.txt).
