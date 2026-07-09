# LCT interactive paper

An interactive research instrument for `lct-activation`: a determinant-one 3D
phase-space explainer, guided learning loop, and progressively disclosed account
of the NanoGPT evaluation.

The interface deliberately separates two things:

- the explorable `M = C(k) S(s) R(theta)` rotation / reciprocal-scale / chirp-shear geometry of the LCT family;
- the fixed Fourier-family transform cells used in the reported NanoGPT arms.

## Run locally

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

## Verify

```bash
npm run lint
npm test
```

`npm test` performs the production build and checks the server-rendered research
paper. The interactive canvas and controls hydrate in the browser.

## Project shape

- `app/page.tsx` contains the 3D canvas, transform controls, learning loop,
  fair-comparison switch, and paper content.
- `app/globals.css` defines the responsive scientific editorial design.
- `tests/rendered-html.test.mjs` protects the finished content and makes sure the
  starter preview cannot return.
- `.openai/hosting.json` retains the Sites hosting declaration.

The result is grounded in `../paper/report.md` and links to the repository's
code and artifacts.
