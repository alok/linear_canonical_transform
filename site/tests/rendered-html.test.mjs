import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the LCT interactive paper", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>LCT — Geometry, Speed, and an Honest Negative Result<\/title>/i);
  assert.match(html, /The transform keeps its area/);
  assert.match(html, /A determinant-one machine/);
  assert.match(html, /Fair comparison is an interaction/);
  assert.match(html, /Structure is real\. Improvement is not\./);
  assert.match(html, /fixed.*linear-fourier/is);
  assert.match(html, /H100 learnable path: implementation passes, quality does not/);
  assert.match(html, /Exploratory \/ debug result/);
  assert.match(html, /1 seed \/ 500 steps/);
  for (const exactValue of [
    "50,782,273",
    "122,151.85",
    "1.7425669",
    "34,021,453",
    "144,000.68",
    "1.9404578",
    "125,735.66",
    "1.9438160",
    "63,751.19",
    "1.9790263",
    "4.3599734438",
    "0.0019682646",
    "1.1920929e-7",
  ]) {
    assert.ok(html.includes(exactValue), `expected rendered H100 value ${exactValue}`);
  }
  assert.match(html, /There is no positive quality claim/);
  assert.match(
    html,
    /https:\/\/github\.com\/alok\/linear_canonical_transform\/blob\/main\/paper\/results\/modal_h100_learnable_s1\.json/,
  );
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("ships the research instrument instead of starter infrastructure", async () => {
  const [page, layout, css, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /<canvas/);
  assert.match(page, /matrixFor/);
  assert.match(page, /determinant/);
  assert.match(page, /M = C\(k\) · S\(s\) · R\(θ\)/);
  assert.match(page, /c: shear \* a - sine \* squeeze/);
  assert.match(page, /d: shear \* b \+ cosine \* squeeze/);
  assert.doesNotMatch(page, /M = R\(θ\) · D\(s\) · H\(k\)/);
  assert.match(page, /CHALLENGES/);
  assert.match(page, /Same width/);
  assert.match(page, /Parameter matched/);
  assert.match(page, /17\.9×/);
  assert.match(page, /4\.3599734438/);
  assert.match(page, /0\.0019682646/);
  assert.match(page, /1\.1920929e-7/);
  assert.match(page, /modal_h100_learnable_s1\.json/);
  assert.match(page, /square attention projections remain untested/i);
  assert.match(css, /prefers-reduced-motion/);
  assert.match(layout, /LCT interactive paper/);
  assert.match(packageJson, /"name": "lct-interactive-paper"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await assert.rejects(access(new URL("public/_sites-preview", templateRoot)));
});
