import { chromium } from "playwright";

const URLS = [
  ["http://localhost:8082/", "01_console.png", 5000],
  ["http://localhost:8082/cycle/25315", "02_drilldown_25315.png", 8000],
];

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
page.on("console", (m) => { if (m.type() === "error") console.error("CONSOLE:", m.text()); });

for (const [url, file, wait] of URLS) {
  console.log("→", url);
  await page.goto(url, { waitUntil: "networkidle" });
  await page.waitForTimeout(wait);
  await page.screenshot({ path: `../shots/${file}`, fullPage: true });
  console.log("  saved", file);
}
await browser.close();
