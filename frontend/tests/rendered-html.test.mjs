import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the MotorLens upload experience", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>MotorLens — AI Vehicle Identifier<\/title>/i);
  assert.match(html, /Identify the car/);
  assert.match(html, /Drop a vehicle photo here/);
  assert.match(html, /Privacy, in plain English/);
  assert.match(html, /Click anywhere to continue/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("keeps consent explicit and supports the five-candidate workflow", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /classification\.predictions\.map/);
  assert.match(page, /Confirm match — don’t store image/);
  assert.match(page, /Contribute this image/);
  assert.match(page, /consent_to_training/);
  assert.match(page, /dismiss(?:al|ing|es).*does not authorize storage/i);
  assert.match(layout, /MotorLens — AI Vehicle Identifier/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});

test("offers automatic, snapped, and painted vehicle selection", async () => {
  const [page, selector] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/VehicleSelector.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /selection_mask/);
  assert.match(selector, /Auto find/);
  assert.match(selector, /Snap outline/);
  assert.match(selector, /Manual brush/);
  assert.match(selector, /Brush size/);
  assert.match(selector, /Eraser/);
  assert.match(selector, /api\/v1\/detect/);
});
