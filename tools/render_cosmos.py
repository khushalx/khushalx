#!/usr/bin/env python3
"""Build both Crimson Cosmos covers from local inputs."""

import base64
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'


def smooth(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def noise(x, y):
    """Continuous, deterministic value noise in art or disk coordinates."""
    i, j = np.floor(x), np.floor(y)
    fx, fy = smooth(0, 1, x - i), smooth(0, 1, y - j)
    i, j = i.astype(np.int32).view(np.uint32), j.astype(np.int32).view(np.uint32)

    def h(dx, dy):
        key = ((i + np.uint32(dx)) * np.uint32(374761393)
               + (j + np.uint32(dy)) * np.uint32(668265263))
        key = (key ^ (key >> 13)) * np.uint32(1274126177)
        return ((key ^ (key >> 16)) & 65535).astype(np.float32) / 65535

    return ((h(0, 0) * (1 - fx) + h(1, 0) * fx) * (1 - fy)
            + (h(0, 1) * (1 - fx) + h(1, 1) * fx) * fy)


def turbulence(x, y, octaves=5):
    out, amplitude = 0, .53
    for _ in range(octaves):
        out += amplitude * noise(x, y)
        x, y = 1.8 * x + .91 * y + 11.4, -.91 * x + 1.8 * y - 8.3
        amplitude *= .48
    return out


def geodesics():
    """Tabulate inverse radius u(phi,b); Schwarzschild units with G=c=M=1.

    Integrate u'' = 3u² - u from a distant camera with u(0)=0 and
    u'(0)=1/b. A ray stops contributing at the horizon or on escape.
    """
    db, dp = .004, .0035
    b = np.arange(.004, 44 + db, db, dtype=np.float64)
    table = np.empty((int(3 * np.pi / dp) + 2, b.size), dtype=np.float32)
    u, v = np.zeros_like(b), 1 / b
    alive = np.ones(b.size, dtype=bool)
    table[0] = u

    def accel(r):
        return r * (3 * r - 1)

    for i in range(1, len(table)):
        k1 = accel(u)
        k2 = accel(u + .5 * dp * v)
        k3 = accel(u + .5 * dp * (v + .5 * dp * k1))
        k4 = accel(u + dp * (v + .5 * dp * k2))
        u += dp * (v + dp / 6 * (k1 + k2 + k3))
        v += dp / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        alive &= (u > 0) & (u < .5)
        table[i] = np.where(alive, u, 0)
        u[~alive], v[~alive] = 0, 0
    return table, dp, db


def disk_light(x, y):
    table, dp, db = geodesics()
    b = np.maximum(np.hypot(x, y), .01)
    nx, ny = x / b, y / b
    # The disk is seen 9° above its equator. Consecutive plane crossings
    # show the direct disk and its gravitationally lensed higher images.
    elevation = np.deg2rad(9)
    se, ce = np.sin(elevation), np.cos(elevation)
    phase = np.mod(np.arctan2(-se, ny * ce), np.pi)
    ib = np.clip(b / db - 1, 0, table.shape[1] - 2.001)
    left = ib.astype(np.int32)
    mix = ib - left
    light, heat = np.zeros_like(x), np.zeros_like(x)
    transmit = np.ones_like(x)

    for order in range(3):
        a = phase + order * np.pi
        ip = a / dp
        lower = ip.astype(np.int32)
        frac = ip - lower
        u1 = table[lower, left] * (1 - mix) + table[lower, left + 1] * mix
        u2 = table[lower + 1, left] * (1 - mix) + table[lower + 1, left + 1] * mix
        r = 1 / np.maximum(u1 * (1 - frac) + u2 * frac, .01)
        sa, ca = np.sin(a), np.cos(a)
        px, py = r * nx * sa, r * (ce * ca - se * ny * sa)
        theta = np.arctan2(py, px)
        # Wind the texture into turbulent shear layers, with no clean
        # concentric outlines. The approaching side is Doppler boosted.
        wind = 2.6 * np.log(r / 6)
        cx, cy = np.cos(theta + wind), np.sin(theta + wind)
        broad = turbulence(cx * 4.5 + r * .37, cy * 4.5 - r * .13, 4)
        fine = turbulence(cx * 11.8 + r * 2.9, cy * 11.8 - r * 1.7, 4)
        wave = .5 + .5 * np.sin(r * 12.4 + broad * 10 + fine * 2.8 + 2 * np.cos(theta + r * .27))
        shreds = .5 + .5 * np.sin(r * 26.5 + broad * 15 + fine * 4)
        filaments = .24 + .76 * (wave ** 2 * .61 + shreds * .12 + fine * .59)
        hot = (np.clip(broad * .77 + fine * .42, 0, 1) * 1.18) ** 2.5
        edge = 5.68 + .37 * np.sin(theta * 3 + .3) + .23 * np.cos(theta * 7)
        outer = 22 + 2.4 * np.sin(2 * theta + .8) + 1.3 * np.cos(5 * theta)
        taper = smooth(edge, edge + 1.3, r) * (1 - smooth(14.5, outer, r))
        speed = 1 / np.sqrt(np.maximum(r - 2, 1))
        doppler = 1 / (1 + .84 * speed * np.cos(theta - .22)) ** 3
        hot_inner = np.exp(-((r - 7.25) / 2.0) ** 2) * (.49 + fine * .75)
        density = taper * np.exp(-(r - 6) * .112) * (filaments * (hot + .25) + hot_inner * .32)
        emission = density * doppler * (1, .92, .64)[order] * transmit
        light += emission
        heat += emission * (hot_inner * .37 + fine * .35 + doppler * .14)
        transmit *= 1 - .91 * taper * smooth(.01, .15, density)

    return b, light, heat


def blur(field, sigma):
    # Separable Gaussian, using an FFT so the dim scattering can stay in float.
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.rfftfreq(field.shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(field) * kernel, s=field.shape, axes=(0, 1)).astype(np.float32)


def artwork():
    """Return desktop/mobile crops of the shared ray-traced exposure."""
    w, h = 1320, 980
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # Camera rotated 13° so the disk cuts the negative space diagonally.
    dx, dy = (xx - 686) / 32, (478 - yy) / 32
    roll = np.deg2rad(13)
    x, y = dx * np.cos(roll) + dy * np.sin(roll), dy * np.cos(roll) - dx * np.sin(roll)
    b, disk, heat = disk_light(x, y)
    ring = np.exp(-((b - 5.208) / .042) ** 2) * .24 * (1 + .81 * np.cos(np.arctan2(y, x) + 2.5))
    ring *= .6 + .5 * noise(x * 3, y * 3)
    disk += ring
    small, mid, far = blur(disk, 2.3), blur(disk, 15), blur(disk, 53)
    core = 1 - np.exp(-1.64 * disk)
    warmth = np.clip((1 - np.exp(-1.25 * heat)) * 1.4, 0, 1)

    # Faint, warped clouds well behind the accretion disk. Mask the event
    # shadow out of the sky before adding optical scatter in front of it.
    warp = turbulence(xx / 150 + 12, yy / 106 - 85, 4)
    sky = turbulence(xx / 211 + warp * 2.7, yy / 94 - warp * 3.4, 6)
    veil = turbulence(x * .32 + warp * 2 + 98, y * .68 - warp * 2, 5)
    plume = np.exp(-((y - .11 * x - 2.0 * np.cos(x * .14)) / (5.5 + .13 * np.abs(x))) ** 2)
    nebula = (sky ** 2.7 * .6 + np.maximum(veil - .20, 0) ** 1.45) * plume
    nebula *= np.exp(-((x + 3) / 24) ** 2) * smooth(5.18, 5.46, b)
    clouds = (warp ** 4) * np.exp(-((dx - 6) / 19) ** 2 - ((dy + 1) / 11) ** 2) * smooth(5.18, 5.46, b)

    # Stretch the wisps into the disk's shear direction. Cloud contrast is
    # masked against the shadow; bloom can cross it as lens scatter.
    strands = turbulence(x * .23 + warp * 4, y * .95 + warp * 1.5, 6)
    wisps = np.maximum(strands - .21, 0) ** 1.42 * plume
    wisps *= np.exp(-((x + 1.5) / 18) ** 2) * smooth(5.14, 6.1, b)
    halo = (1 - np.exp(-far * 4.8)) * .78
    red = 5 + core * 206 + small * 57 + mid * 112 + halo * 130 + nebula * 103 + wisps * 160 + clouds * 20
    green = 5 + core * (8 + warmth * 74) + small * 5 + mid * 9 + halo * 9 + nebula * 4 + wisps * 7
    blue = 7 + core * (11 + warmth * 37) + small * 5 + mid * 10 + halo * 14 + nebula * 12 + wisps * 16
    scatter = np.dstack((red, green, blue))

    rng = np.random.default_rng(79163)
    # Unresolved stars and dust give the exposure fine detail. Keep the
    # interior of the shadow clear and the brightest stars scarce.
    for ix, iy in zip(rng.integers(3, w - 3, 3000), rng.integers(3, h - 3, 3000)):
        if b[iy, ix] < 5.28:
            continue
        strength = rng.random() ** 5
        value = (6 + 147 * strength) * (1 - .55 * core[iy, ix])
        shade = np.array((1, .65, .56) if rng.random() < .53 else (.86, .88, 1)) * value
        for sy in range(-2, 3):
            for sx in range(-2, 3):
                scatter[iy + sy, ix + sx] += shade * np.exp(-(sx * sx + sy * sy) / (.40 + strength))
    grain = rng.normal(0, .74, size=(h, w, 1))
    img = Image.fromarray(np.clip(scatter + grain, 0, 255).astype(np.uint8))
    return img


@lru_cache(maxsize=None)
def font(weight):
    face = TTFont(ROOT / 'tools/fonts/SpaceGrotesk.ttf')
    return instantiateVariableFont(face, {'wght': weight}, inplace=True)


def lettering(label, x, baseline, size, color, weight=650, tracking=0):
    """Outline Space Grotesk glyphs so the artwork needs no browser font load."""
    face = font(weight)
    glyphs, cmap = face.getGlyphSet(), face.getBestCmap()
    scale = size / face['head'].unitsPerEm
    names = [cmap[ord(c)] for c in label]
    advances = [glyphs[name].width * scale + tracking for name in names]
    parts = []
    for name, advance in zip(names, advances):
        pen = SVGPathPen(glyphs, ntos=lambda n: f'{n:.2f}'.rstrip('0').rstrip('.'))
        glyphs[name].draw(pen)
        if pen.getCommands():
            parts.append(f'<path transform="translate({x:.2f} {baseline}) scale({scale:.6f} {-scale:.6f})" d="{pen.getCommands()}"/>')
        x += advance
    return f'<g fill="{color}" aria-label="{label.replace("&", "&amp;")}">{"".join(parts)}</g>'


def picture_payload(img, layout):
    """Crop to the visible frame, then embed one optimized photographic layer."""
    ix, iy, scale = layout['image']
    w, h = layout['size']
    left = max(0, int(-ix / scale))
    top = max(0, int(-iy / scale))
    right = min(img.width, int(np.ceil((w - ix) / scale)))
    bottom = min(img.height, int(np.ceil((h - iy) / scale)))
    buffer = io.BytesIO()
    img.crop((left, top, right, bottom)).save(buffer, 'JPEG', quality=83, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return (f'<image x="{ix + left * scale:.2f}" y="{iy + top * scale:.2f}" '
            f'width="{(right-left)*scale:.2f}" height="{(bottom-top)*scale:.2f}" '
            f'href="data:image/jpeg;base64,{encoded}"/>')


MOTION = '''<style>
@media (prefers-reduced-motion: no-preference) {
  .drift { animation: drift 21s ease-in-out infinite; }
  .flare { animation: flare 11s ease-in-out infinite; }
  .rotor { animation: spin 25s linear infinite; }
  .rotor.slow { animation: spin 38s linear -13s infinite; }
}
@keyframes flare { 0%,100% {opacity:.34} 50% {opacity:.79} }
@keyframes drift { 0%,100% {transform:translate(0,0)} 50% {transform:translate(19px,-7px)} }
@keyframes spin { to {transform:rotate(360deg)} }
:root:target .drift, :root:target .flare, :root:target .rotor { animation:none; }
</style>'''


def sky_marks(layout):
    ix, iy, scale = layout['image']
    cx, cy = ix + 686 * scale, iy + 478 * scale
    rad = 32 * scale
    rng = np.random.default_rng(12229)
    dust = []
    for _ in range(135):
        x = rng.uniform(-rad * 17, rad * 17)
        y = rng.normal(rad * .12, rad * 2.25)
        if x * x + y * y < (rad * 6.3) ** 2:
            continue
        tint = rng.choice(['#FFC3A6', '#FF6E4E', '#FF2637', '#C00A25'])
        dust.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rng.uniform(.44, 1.70):.2f}" '
                    f'fill="{tint}" opacity="{rng.uniform(.25, .73):.2f}"/>')

    def rotor(slow, count, lo, hi):
        sparks = []
        for _ in range(count):
            angle = rng.uniform(0, np.pi * 2)
            distance = rng.uniform(lo, hi) * rad
            x, y = distance * np.cos(angle), distance * np.sin(angle)
            size = rng.uniform(.65, 1.65)
            tint = rng.choice(['#FF543E', '#FF7760', '#FFC5A2'])
            sparks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.2f}" fill="{tint}" '
                          f'opacity="{rng.uniform(.22, .61):.2f}"/>')
        return f'<g class="rotor{slow}">{"".join(sparks)}</g>'

    # Flatten the moving particles into the disk's plane and hide the
    # far-side particles where the black hole's shadow occludes them.
    shadow = rad * 5.15
    return (f'<g transform="translate({cx:.2f} {cy:.2f}) rotate(-13)">'
            f'<defs><mask id="occult" maskUnits="userSpaceOnUse" x="{-21*rad}" y="{-21*rad}" '
            f'width="{42*rad}" height="{42*rad}">'
            f'<rect x="{-21*rad}" y="{-21*rad}" width="{42*rad}" height="{42*rad}" fill="white"/>'
            f'<path d="M{-shadow:.2f} 0A{shadow:.2f} {shadow:.2f} 0 1 1 {shadow:.2f} 0Z" fill="black"/>'
            f'</mask></defs>'
            f'<ellipse class="flare" cx="{-6.3*rad:.1f}" cy="{1.2*rad:.1f}" rx="{6*rad:.1f}" '
            f'ry="{3.3*rad:.1f}" fill="url(#flare-glow)" opacity=".42"/>'
            f'<g mask="url(#occult)"><g class="drift">{"".join(dust)}</g>'
            f'<g transform="translate(0 {1.1*rad:.2f}) scale(1 .19)">'
            f'{rotor("", 52, 5.8, 12)}{rotor(" slow", 44, 10, 17)}</g></g></g>')


def typography(layout):
    mobile = layout['mobile']
    x = 43 if mobile else 73
    name_size = 131 if mobile else 145
    name_y = (147, 248) if mobile else (217, 328)
    subject = '#F1D6D2'
    names = (
        lettering('Khushal', x - 4, name_y[0], name_size, '#F8F1EE', 650, -4.1),
        lettering('Dangar', x - 4, name_y[1], name_size, '#F2DFDA', 650, -4.1),
    )
    subtitle = (
        lettering('Computer Science &', x, 315, 30, subject, 550),
        lettering('Artificial Intelligence', x, 350, 30, subject, 550),
    ) if mobile else (
        lettering('Computer Science & Artificial Intelligence', x, 411, 25, subject, 550),
    )
    return ''.join((*names, *subtitle))


def cover(img, layout):
    w, h = layout['size']
    mobile = layout['mobile']
    axis = 'x2="0" y2="1"' if mobile else ''
    stops = ((0, 1), (.20, 1), (.31, .89), (.46, .69), (.65, 0), (1, 0)) if mobile else (
        (0, 1), (.27, 1), (.43, .64), (.68, 0), (1, 0))
    fade = ''.join(f'<stop offset="{p}" stop-color="#050507" stop-opacity="{a}"/>' for p, a in stops)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" id="still" role="img" aria-labelledby="title desc">
<!-- Generated by tools/render_cosmos.py -->
<title id="title">Khushal Dangar — Crimson Cosmos</title>
<desc id="desc">Computer Science &amp; Artificial Intelligence. Crimson light and glowing dust swirl around a black hole.</desc>
{MOTION}
<defs>
  <linearGradient id="type-shade" {axis}>{fade}</linearGradient>
  <radialGradient id="flare-glow"><stop stop-color="#FF3222" stop-opacity=".62"/><stop offset=".42" stop-color="#D20C22" stop-opacity=".25"/><stop offset="1" stop-color="#9F061A" stop-opacity="0"/></radialGradient>
  <radialGradient id="crimson-sky" cx="65%" cy="76%" r="72%"><stop stop-color="#730B1D" stop-opacity=".72"/><stop offset=".55" stop-color="#460814" stop-opacity=".38"/><stop offset="1" stop-color="#22050B" stop-opacity="0"/></radialGradient>
</defs>
<rect width="{w}" height="{h}" fill="#050507"/>
{picture_payload(img, layout)}
{sky_marks(layout)}
<rect width="{w}" height="{h}" fill="url(#type-shade)"/>
<rect width="{w}" height="{h}" fill="url(#crimson-sky)" opacity=".31"/>
{typography(layout)}
</svg>
'''


def main():
    img = artwork()
    desktop = dict(size=(1280, 526), image=(340, -229, 1.06), mobile=False)
    mobile = dict(size=(620, 730), image=(-184, 141, .80), mobile=True)
    for suffix, layout in (('', desktop), ('-mobile', mobile)):
        (ASSETS / f'crimson-cosmos{suffix}.svg').write_text(cover(img, layout), encoding='utf-8')


if __name__ == '__main__':
    main()
