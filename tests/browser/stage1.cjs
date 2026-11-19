const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const url = process.argv[2] || "http://127.0.0.1:8301/";
const evidence = process.env.PRACTICE_EVIDENCE;
const reference = process.env.PRACTICE_COMPARE_URL;
const welcome = "Study your AI research notes.";

async function noHorizontalOverflow(page) {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
}

async function capture(page, name) {
  if (evidence) await page.screenshot({ path: path.join(evidence, name + ".png"), animations: "disabled" });
}

async function open(page) {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.getByRole("heading", { name: welcome }).waitFor();
}

async function submit(input, prompt, key = "Control+Enter") {
  await input.fill(prompt);
  await input.press(key);
}

async function atEnd(page) {
  const thread = page.getByRole("region", { name: "Conversation" });
  try {
    await page.waitForFunction(() => {
      const thread = document.getElementById("chat-messages");
      return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 3;
    }, null, { timeout: 5000 });
  } catch (error) {
    console.error(await thread.evaluate(el => ({
      top: el.scrollTop, total: el.scrollHeight, client: el.clientHeight,
      focused: document.activeElement?.outerHTML.slice(0, 160),
    })));
    throw error;
  }
  return thread;
}

async function checkDock(page, button) {
  await button.focus();
  await page.waitForFunction(() => {
    const shell = document.querySelector(".practice-app");
    return Math.abs(shell.getBoundingClientRect().height - visualViewport.height) < 1;
  });
  await button.evaluate(el => el.scrollIntoView({ block: "nearest" }));
  const dock = await page.locator(".study-dock").boundingBox();
  const action = await button.boundingBox();
  assert.ok(action.y + action.height <= dock.y + 1, JSON.stringify({ dock, action }));
  assert.ok(action.y >= 0);
}

async function checkLayout(browser) {
  const result = [];
  for (const [width, height] of [
    [1440, 900], [1180, 760], [390, 844], [959, 760], [960, 760], [1279, 760], [1280, 760],
  ]) {
    for (const theme of ["dark", "light"]) {
      for (const reducedMotion of ["no-preference", "reduce"]) {
        const page = await browser.newPage({ viewport: { width, height }, reducedMotion });
        try {
          const suffix = `${width}x${height}-${theme}-${reducedMotion}`;
          await open(page);
          if (theme === "light") await page.getByRole("button", { name: "Switch to light theme" }).click();
          await page.evaluate(() => document.fonts.ready);
          assert.ok(await page.evaluate(() => document.fonts.check('600 40px Archivo')));
          await page.waitForFunction(() => Boolean(document.querySelector(".welcome-diagram")?.dataset.draw));
          const subtitleColor = await page.locator(".welcome-copy > p").evaluate(el => getComputedStyle(el).color);
          const checkPlaceholder = async () => {
            const { color, opacity } = await page.locator("#chat-input").evaluate(el => {
              const style = getComputedStyle(el, "::placeholder");
              return { color: style.color, opacity: Number(style.opacity) };
            });
            assert.equal(color, subtitleColor);
            assert.ok(opacity < 1);
          };
          await checkPlaceholder();
          assert.equal(await page.getByText("Ctrl / ⌘ Enter to ask", { exact: true }).count(), 0);
          assert.equal(await page.locator(".composer-bottom").evaluate(el =>
            Math.abs(el.getBoundingClientRect().right - el.querySelector("button").getBoundingClientRect().right) < 1), true);
          const animation = await page.locator(".welcome-diagram").evaluate(el => {
            const line = el.querySelector(".diagram-line");
            const style = getComputedStyle(line);
            const draw = line.getAnimations().find(item => item.animationName === "connect-concepts");
            return {
              draw: el.dataset.draw,
              name: style.animationName,
              duration: style.animationDuration,
              delay: style.animationDelay,
              initialOffset: parseFloat(draw?.effect.getKeyframes()[0].strokeDashoffset),
              pathLength: Number(line.getAttribute("pathLength")),
            };
          });
          if (reducedMotion === "reduce") {
            assert.equal(animation.draw, "done");
            assert.equal(animation.name, "none");
          } else {
            assert.ok(animation.draw === "done" || animation.name === "connect-concepts");
            if (animation.name === "connect-concepts") {
              assert.equal(animation.duration, "1.75s");
              assert.equal(animation.delay, "0.1s");
              assert.equal(animation.initialOffset, animation.pathLength);
            }
          }
          if ([390, 1440].includes(width) && theme === "light") {
            const diagram = page.locator(".welcome-diagram");
            await page.waitForFunction(() => document.querySelector(".welcome-diagram").dataset.draw === "done");
            await page.locator("#chat-input").fill("Keep this draft");
            await page.getByRole("button", { name: "Switch to dark theme" }).press("Enter");
            assert.equal(await page.locator("#chat-input").inputValue(), "Keep this draft");
            assert.equal(await diagram.getAttribute("data-draw"), reducedMotion === "reduce" ? "done" : "play");
            if (reducedMotion !== "reduce") {
              const animationTime = () => diagram.evaluate(el => el.querySelector(".diagram-line").getAnimations()[0].currentTime);
              await page.waitForFunction(() => document.querySelector(".diagram-line").getAnimations()[0]?.currentTime > 250);
              await page.getByRole("button", { name: "Switch to light theme" }).click();
              assert.ok(await animationTime() < 200);
            } else {
              await page.getByRole("button", { name: "Switch to light theme" }).click();
              assert.equal(await diagram.getAttribute("data-draw"), "done");
            }
          }
          await noHorizontalOverflow(page);
          const captureViewport = reducedMotion === "no-preference";
          if (captureViewport) await capture(page, "app-welcome-" + suffix);
          await submit(page.getByRole("textbox", { name: "Ask about your notes" }), "Layout check");
          const actions = page.getByRole("button", { name: "Copy answer" });
          await actions.waitFor();
          await checkPlaceholder();
          assert.equal(
            await page.locator(".answer-body").evaluate(el =>
              getComputedStyle(el).color === getComputedStyle(document.querySelector(".practice-app")).color),
            true,
          );
          assert.equal(
            await page.locator(".code-block pre").evaluate(el =>
              [...el.querySelectorAll("code, span")].every(token =>
                getComputedStyle(token).backgroundColor === "rgba(0, 0, 0, 0)")),
            true,
          );
          await atEnd(page);
          await checkDock(page, actions);
          const research = page.getByRole("button", { name: "Research", exact: true });
          await research.press("Enter");
          assert.equal(await page.locator("#chat-input").inputValue(), "Research ");
          assert.equal(await page.locator(".user-message").count(), 1);
          assert.equal(await page.locator(".practice-history").isVisible(), width >= 960);
          assert.equal(await page.locator(".practice-history .panel-heading h2").isVisible(), width >= 1280);
          assert.equal(await page.locator(".practice-sources").isVisible(), width >= 960);
          if (width === 1440 && theme === "dark" && reducedMotion === "no-preference") {
            const toggle = page.getByRole("button", { name: "Open sources and concepts" });
            assert.equal(await toggle.count(), 0);
            await page.getByRole("button", { name: "Close sources and concepts" }).click();
            await page.locator(".practice-sources").waitFor({ state: "hidden", timeout: 1500 });
            assert.equal(await toggle.evaluate(el => el === document.activeElement), true);
            await toggle.click();
            await page.locator(".practice-sources").waitFor();
            assert.equal(await page.evaluate(() => document.activeElement.textContent), "Sources & concepts");
          }
          await noHorizontalOverflow(page);
          if (captureViewport) {
            if (evidence) await page.locator(".answer-body .mermaid-wrapper svg").waitFor();
            await page.getByRole("region", { name: "Conversation" }).evaluate(el => {
              el.focus({ preventScroll: true });
              el.scrollTop = 0;
              el.dispatchEvent(new Event("scroll"));
            });
            await capture(page, "app-study-" + suffix);
            if (width === 1440 && theme === "dark") {
              await page.getByRole("button", { name: "Clear session" }).click();
              await page.getByRole("dialog", { name: "Clear this session?" })
                .getByRole("button", { name: "Clear session" }).click();
              await page.getByRole("heading", { name: welcome }).waitFor();
              assert.equal(await page.locator(".diagram-line").evaluate(el => getComputedStyle(el).animationName), "none");
            }
            if (reference && evidence) {
              await page.goto(`${reference}?theme=${theme}#welcome`, { waitUntil: "domcontentloaded" });
              await page.evaluate(() => document.fonts.ready);
              await page.waitForFunction(() => document.querySelector(".welcome-diagram")?.dataset.intro === "done");
              await capture(page, "reference-welcome-" + suffix);
              await page.goto(`${reference}?theme=${theme}#study`, { waitUntil: "domcontentloaded" });
              await page.waitForFunction(() => document.body.dataset.screen === "study");
              await page.evaluate(() => document.fonts.ready);
              await capture(page, "reference-study-" + suffix);
            }
          }
          result.push(suffix);
        } finally {
          await page.close();
        }
      }
    }
  }
  return result;
}

async function checkInteractions(browser) {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    hasTouch: true,
    isMobile: true,
    reducedMotion: "reduce",
    permissions: ["clipboard-read", "clipboard-write"],
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    // Reflex 0.8.28.post1's generated Socket.IO client also emitted this in Stage 0.
    if (message.type() === "error" &&
        message.text() !== "Permissions policy violation: unload is not allowed in this document.") {
      errors.push(message.text());
    }
  });
  await open(page);
  const input = page.getByRole("textbox", { name: "Ask about your notes" });
  await page.getByRole("button", { name: /Quiz me on Diffusions/ }).click();
  assert.equal(await input.inputValue(), "Quiz me on Diffusions topic");
  assert.equal(await input.evaluate(el => el === document.activeElement), true);
  await input.press("Shift+Enter");
  assert.ok((await input.inputValue()).endsWith("\n"));
  await page.evaluate(() => {
    const seen = { math: false, mermaid: false };
    const observer = new MutationObserver(() => {
      const partial = document.querySelector(".streaming-message");
      seen.math ||= (partial?.textContent?.includes("$O(") && !partial.querySelector(".katex"));
      seen.mermaid ||= Boolean(partial?.querySelector(".mermaid-wrapper pre")?.textContent.includes("Input -->"));
    });
    observer.observe(document.querySelector(".practice-app"), { subtree: true, childList: true, characterData: true });
    window.readPracticeStream = () => { observer.disconnect(); return seen; };
  });
  await submit(input, "Explain KV caching");
  await page.locator(".streaming-message").waitFor();
  const responseId = await page.locator(".streaming-message").getAttribute("data-response-id");
  assert.match(responseId, /^[0-9a-f-]{36}$/);
  const history = page.getByRole("button", { name: "Open current-session history" });
  await history.click();
  const drawer = page.getByRole("dialog", { name: "Current-session history" });
  assert.equal(await drawer.getByRole("button", { name: "Clear session" }).isEnabled(), true);
  await drawer.getByRole("button", { name: /Explain KV caching/ }).click();
  await page.waitForFunction(() => document.activeElement.classList.contains("user-message"));
  const firstAnswer = page.locator(".assistant-message").first();
  await firstAnswer.getByRole("button", { name: "Copy answer" }).waitFor();
  assert.equal(await firstAnswer.getAttribute("data-response-id"), responseId);
  assert.deepEqual(await page.evaluate(() => window.readPracticeStream()), { math: true, mermaid: true });
  const thread = page.getByRole("region", { name: "Conversation" });
  assert.ok(await thread.evaluate(el => el.scrollTop < 50));
  assert.ok(await firstAnswer.locator("table").isVisible());
  assert.ok(await firstAnswer.locator(".katex-display").isVisible());
  await firstAnswer.locator(".mermaid-wrapper svg").waitFor({ timeout: 10000 });
  assert.ok(await firstAnswer.locator(".code-block pre").evaluate(el => el.scrollWidth > el.clientWidth));
  await firstAnswer.getByRole("button", { name: "Copy code" }).click();
  assert.match(await page.evaluate(() => navigator.clipboard.readText()), /^cache = model\.allocate_cache\(\)\nreview = /);
  await firstAnswer.getByRole("button", { name: "Copy answer" }).click();
  assert.match(await page.evaluate(() => navigator.clipboard.readText()), /What the cache stores/);
  await checkDock(page, firstAnswer.getByRole("button", { name: "Copy answer" }));

  const followUp = page.getByRole("textbox", { name: "Ask a follow-up" });
  await followUp.fill("Draft to keep");
  await page.getByRole("button", { name: "Graph", exact: true }).click();
  await page.getByRole("button", { name: "Study", exact: true }).click();
  await followUp.waitFor();
  assert.equal(await followUp.inputValue(), "Draft to keep");

  const sources = firstAnswer.getByRole("button", { name: "Sources & concepts" });
  await sources.click();
  const sourceDrawer = page.getByRole("dialog", { name: "Sources and concepts" });
  await sourceDrawer.getByText("Source details are unavailable for this response.").waitFor();
  await page.keyboard.press("Escape");
  assert.equal(await sources.evaluate(el => el === document.activeElement), true);
  await history.click();
  await drawer.getByRole("button", { name: "Clear session" }).click();
  const confirmation = page.getByRole("dialog", { name: "Clear this session?" });
  await confirmation.getByRole("button", { name: "Keep studying" }).waitFor();
  assert.equal(await confirmation.getByRole("button", { name: "Keep studying" }).evaluate(el => el === document.activeElement), true);
  await page.keyboard.press("Escape");
  assert.equal(await drawer.getByRole("button", { name: "Clear session" }).evaluate(el => el === document.activeElement), true);
  await page.keyboard.press("Escape");
  assert.equal(await history.evaluate(el => el === document.activeElement), true);

  await history.click();
  await page.touchscreen.tap(8, 150);
  await drawer.waitFor({ state: "hidden" });
  assert.equal(await history.evaluate(el => el === document.activeElement), true);

  await history.click();
  await page.setViewportSize({ width: 960, height: 760 });
  await drawer.waitFor({ state: "hidden" });
  assert.equal(await page.getByRole("button", { name: "Toggle current-session history" }).evaluate(el => el === document.activeElement), true);
  await page.setViewportSize({ width: 844, height: 390 });
  await page.getByRole("button", { name: "Quiz me" }).click();
  assert.equal(await followUp.inputValue(), "Quiz me on ");
  await followUp.fill("Expanded draft\n".repeat(18));
  const expanded = await followUp.boundingBox();
  assert.ok(expanded.height > 28 && expanded.height <= 130);
  await checkDock(page, firstAnswer.getByRole("button", { name: "Copy answer" }));
  await noHorizontalOverflow(page);
  await page.setViewportSize({ width: 390, height: 844 });
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setSafeAreaInsetsOverride", { insets: { top: 12, bottom: 34 } });
  assert.equal(await page.locator(".study-dock").evaluate(el => getComputedStyle(el).paddingBottom), "34px");
  assert.equal(await page.locator(".practice-header").evaluate(el => getComputedStyle(el).paddingTop), "12px");
  await followUp.focus();
  // Chromium can emulate a smaller visual viewport with the same layout dimensions.
  // Headless runs do not open an operating-system keyboard.
  await page.evaluate(() => {
    Object.defineProperties(visualViewport, {
      height: { configurable: true, value: 470 },
      offsetTop: { configurable: true, value: 26 },
    });
    visualViewport.dispatchEvent(new Event("resize"));
  });
  await checkDock(page, firstAnswer.getByRole("button", { name: "Copy answer" }));
  const dock = await page.locator(".study-dock").boundingBox();
  assert.equal(Math.round(dock.y + dock.height), 496);
  await page.evaluate(() => {
    delete visualViewport.height;
    delete visualViewport.offsetTop;
    visualViewport.dispatchEvent(new Event("resize"));
  });
  await checkDock(page, firstAnswer.getByRole("button", { name: "Copy answer" }));
  await cdp.send("Emulation.setSafeAreaInsetsOverride", { insets: {} });
  await cdp.send("Emulation.setPageScaleFactor", { pageScaleFactor: 1.25 });
  await page.waitForFunction(() => visualViewport.scale > 1);
  await noHorizontalOverflow(page);
  await cdp.send("Emulation.resetPageScaleFactor");
  await page.waitForFunction(() => visualViewport.scale === 1);
  await cdp.detach();
  await submit(followUp, "Quiz me on KV caching", "Enter");
  await page.getByText("Practice guidance", { exact: true }).last().waitFor();
  await page.getByRole("button", { name: "Copy answer" }).nth(1).waitFor();
  await atEnd(page);
  await submit(followUp, "Reject the next turn");
  await page.getByRole("button", { name: "Copy answer" }).nth(2).waitFor();
  await followUp.evaluate(el => new Promise(resolve => {
    if (!el.readOnly) return resolve();
    const observer = new MutationObserver(() => {
      if (!el.readOnly) { observer.disconnect(); resolve(); }
    });
    observer.observe(el, { attributes: true });
  }));
  await submit(followUp, "Rejected question");
  await page.getByText("Submission failed").waitFor();
  assert.equal(await page.getByText("Rejected question").count(), 0);
  await submit(followUp, "Map the cache");
  await page.getByRole("button", { name: "Open map: Cache map" }).click();
  await page.locator(".balanced-graph [data-node]").first().waitFor();
  await page.getByRole("button", { name: "Study", exact: true }).click();
  await thread.locator(".user-message").filter({ hasText: "Map the cache" }).waitFor();
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByRole("textbox", { name: "Ask a follow-up" }).waitFor();
  await page.getByRole("button", { name: "Switch to dark theme" }).waitFor();
  const [oldSession, visitor] = await page.evaluate(() => [
    sessionStorage.getItem("ai_practice_session_id"),
    localStorage.getItem("ai_practice_visitor_id"),
  ]);
  assert.ok(oldSession && visitor);
  assert.equal(await firstAnswer.getAttribute("data-response-id"), responseId);
  await page.getByRole("button", { name: "Open current-session history" }).click();
  await drawer.getByRole("button", { name: /Explain KV caching/ }).click();
  await page.waitForFunction(() => document.activeElement.matches(".user-message"));
  await page.getByRole("button", { name: "Open current-session history" }).click();
  await drawer.getByRole("button", { name: /Cache map/ }).click();
  await page.locator(".balanced-graph [data-node]").first().waitFor();
  await page.getByRole("button", { name: "Study", exact: true }).click();

  const newTab = await context.newPage();
  await open(newTab);
  await newTab.waitForFunction(() => Boolean(sessionStorage.getItem("ai_practice_session_id")));
  const [newSession, sameVisitor] = await newTab.evaluate(() => [
    sessionStorage.getItem("ai_practice_session_id"),
    localStorage.getItem("ai_practice_visitor_id"),
  ]);
  assert.notEqual(newSession, oldSession);
  assert.equal(sameVisitor, visitor);
  await newTab.close();

  await page.getByRole("button", { name: "Open current-session history" }).click();
  await drawer.getByRole("button", { name: "Clear session" }).click();
  await confirmation.getByRole("button", { name: "Clear session" }).click();
  await page.getByRole("heading", { name: welcome }).waitFor();
  await page.waitForFunction(previous => sessionStorage.getItem("ai_practice_session_id") !== previous, oldSession);
  assert.equal(await page.evaluate(() => localStorage.getItem("ai_practice_visitor_id")), visitor);
  assert.equal(await input.evaluate(el => el === document.activeElement), true);
  assert.equal(await page.locator(".diagram-line").evaluate(el => getComputedStyle(el).animationName), "none");
  await page.getByRole("button", { name: /Quiz me on Diffusions/ }).click();
  await input.press("Control+Enter");
  await page.getByText("Cached practice question.", { exact: true }).waitFor();

  const restoredInput = page.getByRole("textbox", { name: "Ask a follow-up" });
  await submit(restoredInput, "Reload this streaming turn");
  const reloadStream = page.getByText("Reload test in progress", { exact: true });
  await reloadStream.waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByText("Reload test completed", { exact: true }).waitFor();
  assert.equal(await page.getByText("Reload test completed", { exact: true }).count(), 1);
  await submit(restoredInput, "Clear this streaming turn");
  await page.getByText("Clear test in progress", { exact: true }).waitFor();
  const activeSession = await page.evaluate(() => sessionStorage.getItem("ai_practice_session_id"));
  await page.getByRole("button", { name: "Open current-session history" }).click();
  await drawer.getByRole("button", { name: "Clear session" }).click();
  await confirmation.getByRole("button", { name: "Clear session" }).click();
  await page.getByRole("heading", { name: welcome }).waitFor();
  await page.waitForFunction(previous => sessionStorage.getItem("ai_practice_session_id") !== previous, activeSession);
  assert.equal(await page.getByText("Late clear test answer", { exact: true }).count(), 0);
  assert.equal(await page.evaluate(() => localStorage.getItem("ai_practice_visitor_id")), visitor);
  assert.deepEqual(errors, []);
  await context.close();
}

async function waitForDisclosure(page, details, body) {
  await page.waitForFunction(([el, selector]) => {
    const content = el.querySelector(selector);
    return el.open && content?.getBoundingClientRect().bottom <= el.getBoundingClientRect().bottom + 1;
  }, [await details.elementHandle(), body]);
}

async function reloadFromSnapshot(page) {
  // Reflex 0.8.28.post1 reuses this framework token across ordinary reloads.
  await page.evaluate(() => sessionStorage.removeItem("token"));
  await page.reload();
}

async function checkStudyHistory(browser) {
  for (const width of [1440, 390]) {
    const mobile = width < 960;
    const page = await browser.newPage({ viewport: { width, height: 844 } });
    try {
      const history = page.locator(mobile ? "#history-dialog" : ".practice-history");
      const items = history.locator('button[data-practice-action="history-link"]');
      const prompts = page.locator(".study-turn .user-message");
      const thread = page.getByRole("region", { name: "Conversation" });
      const openHistory = async () => {
        if (mobile) await page.getByRole("button", { name: "Open current-session history" }).click();
      };
      const assertFrame = async () => {
        const frame = await page.evaluate(() => ({
          header: document.querySelector(".practice-header").getBoundingClientRect().top,
          dock: document.querySelector(".study-dock").getBoundingClientRect().bottom,
          height: window.innerHeight,
          shellScroll: document.querySelector(".practice-app").scrollTop,
          pageScroll: window.scrollY,
        }));
        assert.ok(Math.abs(frame.header) < 1 && Math.abs(frame.dock - frame.height) < 1, JSON.stringify(frame));
        assert.equal(frame.shellScroll, 0);
        assert.equal(frame.pageScroll, 0);
      };
      const expectActive = async index => {
        await page.waitForFunction(({ mobile, index }) => {
          const panel = document.querySelector(mobile ? "#history-dialog" : ".practice-history");
          const turns = [...panel.querySelectorAll('button[data-practice-action="history-link"]')];
          return turns.filter(turn => turn.getAttribute("aria-current") === "location").length === 1
            && turns[index]?.getAttribute("aria-current") === "location";
        }, { mobile, index }, { timeout: 4000 });
      };

      await open(page);
      const input = page.locator("#chat-input");
      await input.focus();
      await page.waitForFunction(() => {
        const input = document.querySelector("#chat-input");
        const composer = getComputedStyle(input.closest(".practice-composer"));
        return composer.borderColor === getComputedStyle(input).caretColor;
      });
      assert.deepEqual(await input.evaluate(el => {
        const style = getComputedStyle(el);
        return [style.outlineStyle, style.boxShadow];
      }), ["none", "none"]);
      for (const [index, message] of ["Layout check", "Layout check", "Study without retrieval"].entries()) {
        await submit(input, message);
        if (index === 0) {
          await page.waitForFunction(() => document.activeElement?.id === "chat-messages");
          assert.deepEqual(await thread.evaluate(el => {
            const style = getComputedStyle(el);
            return [style.outlineStyle, style.boxShadow];
          }), ["none", "none"]);
        }
        await page.getByRole("button", { name: "Copy answer" }).nth(index).waitFor();
        await page.waitForFunction(() => !document.querySelector("#chat-input").readOnly);
      }
      await assertFrame();
      await thread.evaluate(element => { element.scrollTop = 0; });
      await expectActive(0);
      await openHistory();
      assert.deepEqual(await history.locator(".sidebar-history-list").first().evaluate(el => {
        const style = getComputedStyle(el);
        const weights = [...el.querySelectorAll(".sidebar-history-title")].map(title => getComputedStyle(title).fontWeight);
        return [style.borderTopWidth, style.gap, weights];
      }), ["0px", "7px", ["600", "400", "400"]]);
      for (const index of [1, 0]) {
        if (index === 0) await openHistory();
        await items.nth(index).click();
        await page.waitForFunction(index => document.activeElement === document.querySelectorAll(".user-message")[index], index);
        await expectActive(index);
        await assertFrame();
      }

      await prompts.nth(1).evaluate(prompt => {
        const scrollport = prompt.closest("#chat-messages");
        scrollport.scrollTop += prompt.getBoundingClientRect().top - scrollport.getBoundingClientRect().top + 5;
      });
      await expectActive(1);
      await assertFrame();
      await thread.evaluate(element => { element.scrollTop = element.scrollHeight; });
      await expectActive(2);

      await openHistory();
      await items.nth(1).click();
      await page.waitForFunction(() => document.activeElement === document.querySelectorAll(".user-message")[1]);
      await expectActive(1);
      await assertFrame();
      await capture(page, `study-history-${width}`);
    } finally {
      await page.close();
    }
  }
}

async function checkStage3(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  let delaySourceSelection = false;
  let releaseSourceSelection;
  await page.routeWebSocket(url => url.pathname.startsWith("/_event"), socket => {
    const backend = socket.connectToServer();
    socket.onMessage(message => {
      if (delaySourceSelection && String(message).includes("select_response_sources")) {
        releaseSourceSelection = () => backend.send(message);
      } else {
        backend.send(message);
      }
    });
  });
  page.on("pageerror", error => errors.push(error.message));
  try {
    await open(page);
    const panel = page.getByRole("complementary", { name: "Sources and concepts" });
    const answer = text => page.locator(".assistant-message").filter({ hasText: text });
    const selectSources = text => answer(text).getByRole("button", { name: "Sources & concepts" });
    const thread = page.getByRole("region", { name: "Conversation" });
    const followUp = page.getByRole("textbox", { name: "Ask a follow-up" });
    const ask = async prompt => {
      await followUp.waitFor();
      await page.waitForFunction(() => !document.querySelector("#chat-input")?.readOnly);
      await submit(followUp, prompt);
    };
    const turn = prompt => thread.locator(".study-turn").filter({
      has: page.locator(".user-message").filter({ hasText: prompt }),
    });
    const executionAction = prompt => turn(prompt).getByRole("button", { name: "Execution details" });
    const openExecution = async (prompt, status, keyboard = false) => {
      const trigger = executionAction(prompt);
      if (keyboard) {
        await trigger.focus();
        await trigger.press("Enter");
      } else {
        await trigger.click();
      }
      const panel = turn(prompt).getByRole("region", { name: "Execution details" });
      await panel.locator(".execution-badge").getByText(status, { exact: true }).waitFor();
      assert.equal(await trigger.getAttribute("aria-expanded"), "true");
      assert.equal(await trigger.getAttribute("aria-controls"), await panel.getAttribute("id"));
      assert.ok((await panel.boundingBox()).y > (await trigger.boundingBox()).y);
      return panel;
    };
    const outcome = async (prompt, status) => {
      const details = await openExecution(prompt, status);
      await executionAction(prompt).click();
      await details.waitFor({ state: "hidden" });
    };
    const checkSourcePreview = async (container) => {
      const list = container.getByRole("list");
      const rows = list.getByRole("listitem");
      assert.equal(await rows.count(), 5);
      const more = container.getByRole("button", { name: "Show all (8)" });
      assert.equal(await more.getAttribute("aria-expanded"), "false");
      assert.equal(await more.getAttribute("aria-controls"), await list.getAttribute("id"));
      await more.press("Enter");
      await list.getByText("Supplemental Study note 8", { exact: true }).waitFor();
      assert.equal(await rows.count(), 8);
      const fewer = container.getByRole("button", { name: "Show fewer" });
      assert.equal(await fewer.getAttribute("aria-expanded"), "true");
      await fewer.press("Enter");
      assert.equal(await rows.count(), 5);
      assert.equal(await more.evaluate(el => el === document.activeElement), true);
    };

    await submit(page.getByRole("textbox", { name: "Ask about your notes" }), "Study local notes");
    await panel.getByRole("status").filter({ hasText: "Source details are pending." }).waitFor({ timeout: 4000 });
    const progress = page.getByRole("list", { name: "Agent progress" });
    await progress.getByText("Analyst", { exact: true }).locator("..").filter({ hasText: "Active" }).waitFor();
    const progressBox = await progress.boundingBox();
    const composerBox = await page.locator(".practice-composer").boundingBox();
    const processingLine = await page.locator(".processing-status").boundingBox();
    assert.ok(Math.abs(processingLine.x - composerBox.x) <= 2);
    assert.ok(progressBox.x > processingLine.x + processingLine.width);
    assert.ok(Math.abs(progressBox.y - processingLine.y) <= 3);
    assert.equal(await progress.locator('[data-outcome="running"] .agent-marker').evaluate(
      el => getComputedStyle(el).animationName
    ), "agent-pulse");
    assert.deepEqual(await progress.locator('[data-outcome="running"] .agent-marker').evaluate(el => {
      const dot = getComputedStyle(el, "::before");
      return [dot.borderTopWidth, dot.outlineStyle];
    }), ["0px", "none"]);
    const composerAlignment = await page.locator(".practice-composer").evaluate(el => {
      const input = el.querySelector("textarea");
      const label = document.createRange();
      label.selectNodeContents(el.querySelector(".intent-controls button"));
      return Math.abs(input.getBoundingClientRect().left - label.getBoundingClientRect().left);
    });
    assert.ok(composerAlignment <= 1);
    assert.equal(await page.locator("#chat-input").getAttribute("placeholder"), "Response in progress…");
    assert.equal(await page.locator("#chat-input").evaluate(el => el.readOnly), true);
    assert.equal(await page.getByRole("button", { name: "Ask", exact: true }).isDisabled(), true);
    assert.equal(await executionAction("Study local notes").count(), 0);
    await capture(page, "app-processing-desktop-dark");
    const firstSourceButton = selectSources("Local explanation one");
    const note = panel.getByText("Decoder attention notes", { exact: true });
    await page.locator(".streaming-message").getByText("Second response in progress").waitFor();
    assert.deepEqual(await progress.locator("li").evaluateAll(elements =>
      elements.map(el => [el.firstElementChild.nextElementSibling.textContent, el.dataset.outcome])), [
      ["Orchestrator", "completed"], ["Retriever", "completed"], ["Researcher", "inactive"],
      ["Analyst", "running"], ["Mentor", "inactive"], ["Visualizer", "inactive"],
    ]);
    assert.deepEqual(await progress.boundingBox(), progressBox);
    assert.equal(await executionAction("Study local notes").count(), 0);
    await panel.getByRole("status").filter({ hasText: "Source details are pending." }).waitFor();
    await firstSourceButton.click();
    await note.waitFor();
    await answer("Local explanation two").waitFor();
    assert.deepEqual(await answer("Local explanation two").evaluate(el => [
      getComputedStyle(el.querySelector(".answer-meta")).borderBottomWidth,
      getComputedStyle(el.querySelector(".answer-body")).borderBottomWidth,
    ]), ["1px", "1px"]);
    await note.waitFor();
    await selectSources("Local explanation two").click();
    await panel.getByText("Grouped query notes", { exact: true }).waitFor();
    assert.equal(await panel.getByRole("button", { name: /^Show all/ }).count(), 0);
    delaySourceSelection = true;
    await firstSourceButton.click();
    await panel.getByRole("status").filter({ hasText: "Source details are pending." }).waitFor({ timeout: 1500 });
    assert.equal(await panel.getByText("Grouped query notes", { exact: true }).isVisible(), false);
    delaySourceSelection = false;
    await page.waitForFunction(() => document.activeElement.matches(".practice-sources h2"));
    releaseSourceSelection();
    await note.waitFor();
    assert.equal(await page.evaluate(() => document.activeElement.textContent), "Sources & concepts");
    assert.equal(await panel.locator(".source-preview").count(), 0);
    assert.equal(await panel.getByText("Transformers / Decoder", { exact: true }).count(), 0);
    assert.equal(await panel.getByText("Retrieval score 0.87", { exact: true }).count(), 0);
    await checkSourcePreview(panel);
    await capture(page, "app-source-preview-desktop-dark");
    await note.click();
    await panel.locator(".source-body").getByText("Explains how the decoder caches attention projections.", { exact: true }).waitFor();
    assert.equal(await panel.getByText("Chunk summary", { exact: true }).count(), 0);
    assert.equal(await panel.getByText("The decoder retains key and value vectors.").count(), 0);
    await panel.locator(".source-note[open] summary").getByText("Summary", { exact: true }).waitFor();
    assert.equal(await panel.getByText("Note details unavailable", { exact: true }).count(), 0);
    assert.equal(await panel.getByText("No retained map contains these concepts.", { exact: true }).count(), 0);
    assert.equal(await panel.getByRole("link").count(), 0);
    assert.equal(await firstSourceButton.getAttribute("aria-pressed"), "true");
    await executionAction("Study local notes").waitFor();
    await capture(page, "app-study-actions-desktop-dark");
    const localExecution = await openExecution("Study local notes", "Answered", true);
    assert.deepEqual(await localExecution.locator(".stage-label strong").allTextContents(),
      ["Orchestrator", "Retriever", "Orchestrator", "Analyst", "Orchestrator", "Analyst"]);
    assert.deepEqual(await localExecution.locator(".stage-duration").allTextContents(),
      Array(6).fill("0.25 s"));
    await page.evaluate(() => document.fonts.ready);
    if (evidence) {
      await waitForDisclosure(page, panel.locator(".source-list details").first(), ".source-body");
    }
    await capture(page, "app-local-desktop-dark");
    await executionAction("Study local notes").press("Enter");
    await localExecution.waitFor({ state: "hidden" });
    assert.equal(await executionAction("Study local notes").evaluate(el => el === document.activeElement), true);
    if (evidence) {
      await page.getByRole("button", { name: "Switch to light theme" }).click();
      await capture(page, "app-study-actions-desktop-light");
      await page.getByRole("button", { name: "Switch to dark theme" }).click();
    }

    await ask("Study web fallback");
    await answer("Web fallback explanation").waitFor();
    const web = panel.getByRole("link", { name: "Open Web research context" });
    await panel.getByText("Web research context", { exact: true }).click();
    await web.waitFor();
    assert.equal(await web.getAttribute("href"), "https://example.org/reference");
    assert.equal(await panel.getByText("Sources supplied as context for this response.", { exact: true }).count(), 0);
    assert.equal(await panel.getByText("Excerpt unavailable.", { exact: true }).count(), 0);
    await panel.getByText("Web record without a link", { exact: true }).waitFor();
    assert.equal(await panel.getByText("Source link unavailable.", { exact: true }).count(), 0);
    assert.equal(await panel.getByText("Excerpt unavailable.", { exact: true }).count(), 0);
    assert.equal(await note.count(), 0);
    const responseId = await answer("Local explanation one").getAttribute("data-response-id");
    await firstSourceButton.click();
    await note.waitFor();
    await page.waitForFunction(() => !document.querySelector("#chat-input")?.readOnly);
    await reloadFromSnapshot(page);
    await firstSourceButton.click();
    await note.waitFor();
    assert.equal(await answer("Local explanation one").getAttribute("data-response-id"), responseId);
    await outcome("Study local notes", "Answered");
    const webExecution = await openExecution("Study web fallback", "Answered");
    assert.deepEqual(await webExecution.locator(".stage-label strong").allTextContents(),
      ["Orchestrator", "Retriever", "Orchestrator", "Researcher", "Analyst"]);
    await executionAction("Study web fallback").click();
    await page.getByRole("button", { name: /Study local notes/ }).click();
    await panel.getByText("Grouped query notes", { exact: true }).waitFor();
    await page.waitForFunction(() => document.activeElement.matches(".user-message"));
    await panel.getByText("Grouped query notes", { exact: true }).click();
    await panel.getByText("Summary unavailable.", { exact: true }).waitFor();
    assert.equal(await panel.getByText("Query heads can share key and value heads.").count(), 0);

    await ask("Study without retrieval");
    await answer("Mentor practice reply").waitFor();
    await panel.getByText("Source details are unavailable for this response.").waitFor();
    await outcome("Study without retrieval", "Answered");
    await ask("Study with empty evidence");
    await panel.getByText("No source details to display for this response.").waitFor();
    assert.equal(await panel.getByRole("heading", { name: "Notes used as context" }).count(), 0);
    await ask("Study relationship notes");
    const relationshipNotes = panel.locator("section").filter({
      has: page.getByRole("heading", { name: "Notes used as context" }),
    });
    const summaryPreview = relationshipNotes.locator(".source-preview");
    await summaryPreview.waitFor();
    assert.match(await summaryPreview.textContent(), /^Attention projections map each token/);
    assert.ok(await summaryPreview.evaluate(el => el.scrollHeight > el.clientHeight));
    assert.equal(await summaryPreview.evaluate(el => el === el.parentElement.firstElementChild), true);
    assert.equal(await relationshipNotes.getByText("Note 1", { exact: true }).count(), 0);
    assert.equal(await relationshipNotes.getByText("Summary", { exact: true }).isVisible(), false);
    assert.equal(await relationshipNotes.evaluate(el =>
      parseFloat(getComputedStyle(el.querySelector("h3")).fontSize) >
      parseFloat(getComputedStyle(el.querySelector("summary")).fontSize)), true);
    await capture(page, "app-relationship-notes-desktop-dark");
    await summaryPreview.click();
    const fullSummary = relationshipNotes.locator(".source-body p").first();
    await fullSummary.waitFor();
    assert.equal(await relationshipNotes.getByText("Chunk summary", { exact: true }).count(), 0);
    assert.match(await fullSummary.textContent(), /combine information from the value vectors\.$/);
    assert.equal(await summaryPreview.isVisible(), false);
    await waitForDisclosure(page, relationshipNotes.locator("details"), ".source-body");
    await capture(page, "app-relationship-summary-expanded-desktop-dark");
    await ask("Study relationships without notes");
    await panel.getByText("The Retriever supplied relationships, but their supporting note excerpts are unavailable.").waitFor();
    await ask("Refuse outside scope");
    await outcome("Refuse outside scope", "Outside study scope");
    await panel.getByText("Source details are unavailable for this response.").waitFor();
    await ask("Finish with no results");
    await outcome("Finish with no results", "No results");
    await panel.getByText("This turn has no response with source details.").waitFor();
    await ask("Fail after one response");
    await progress.locator('li[data-outcome="failed"]').getByText("Researcher").waitFor();
    const failure = await openExecution("Fail after one response", "Failed");
    await failure.locator(".stage-status").getByText("Failed", { exact: true }).waitFor();
    await executionAction("Fail after one response").click();
    await selectSources("Completed before failure").click();
    await note.waitFor();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await ask("Interrupt a stream");
    await progress.locator('li[data-outcome="running"]').getByText("Analyst").waitFor();
    assert.equal(await progress.locator('[data-outcome="running"] .agent-marker').evaluate(
      el => getComputedStyle(el).animationName
    ), "none");
    assert.equal(await page.locator(".composer-bottom").evaluate(el => {
      const { top, bottom } = el.getBoundingClientRect();
      return top >= visualViewport.offsetTop && bottom <= visualViewport.offsetTop + visualViewport.height;
    }), true);
    await capture(page, "app-processing-mobile-light");
    const interrupted = await openExecution("Interrupt a stream", "Interrupted");
    await interrupted.locator(".stage-status").getByText("Interrupted", { exact: true }).waitFor();
    await executionAction("Interrupt a stream").click();
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.emulateMedia({ reducedMotion: "no-preference" });
    assert.equal(await turn("Interrupt a stream").locator(".assistant-message").count(), 0);

    await ask("Map the cache");
    const graph = await openExecution("Map the cache", "Map ready");
    const artifact = turn("Map the cache").getByRole("button", { name: "Open map: Cache map" });
    await graph.getByText("No agent invocations were recorded.").waitFor();
    await executionAction("Map the cache").click();
    assert.equal(await turn("Map the cache").getByRole("button", { name: /^Open map:/ }).count(), 1);
    assert.equal(await page.getByText("Knowledge graph updated from this session").count(), 0);
    await artifact.click();
    await page.locator(".balanced-graph [data-node]").first().waitFor();
    await page.getByRole("button", { name: "Study", exact: true }).click();

    await ask("Study without an execution report");
    await outcome("Study without an execution report", "Answered");
    await page.waitForFunction(() => !document.querySelector("#chat-input")?.readOnly);
    await reloadFromSnapshot(page);
    const unavailable = await openExecution("Study without an execution report", "Unavailable");
    await unavailable.getByText("Execution details are unavailable for this turn.").waitFor();
    await executionAction("Study without an execution report").click();

    await page.setViewportSize({ width: 390, height: 844 });
    await firstSourceButton.click();
    const drawer = page.getByRole("dialog", { name: "Sources and concepts" });
    await drawer.getByText("Decoder attention notes", { exact: true }).waitFor();
    await checkSourcePreview(drawer);
    await capture(page, "app-source-preview-mobile-light");
    await drawer.getByText("Decoder attention notes", { exact: true }).click();
    await drawer.locator(".source-body").getByText("Explains how the decoder caches attention projections.").waitFor();
    assert.equal(await drawer.getByRole("heading", { name: "Related concepts" }).count(), 0);
    await noHorizontalOverflow(page);
    if (evidence) {
      await waitForDisclosure(page, drawer.locator(".source-list details").first(), ".source-body");
    }
    await capture(page, "app-local-mobile-light");
    await page.keyboard.press("Escape");
    assert.equal(await firstSourceButton.evaluate(el => el === document.activeElement), true);
    if (evidence) {
      await page.getByRole("button", { name: "Switch to dark theme" }).click();
      await firstSourceButton.click();
      const noteDisclosure = drawer.locator(".source-note").first();
      await noteDisclosure.waitFor();
      if (!await noteDisclosure.evaluate(el => el.open)) {
        await noteDisclosure.locator("summary").click();
      }
      await waitForDisclosure(page, noteDisclosure, ".source-body");
      await capture(page, "app-local-mobile-dark");
      await page.keyboard.press("Escape");
    }

    await page.setViewportSize({ width: 1440, height: 900 });
    const darkTheme = page.getByRole("button", { name: "Switch to dark theme" });
    if (await darkTheme.isVisible()) await darkTheme.click();
    await ask("Compare agent colors");
    await progress.locator('li[data-outcome="running"]').getByText("Visualizer").waitFor();
    assert.equal(await progress.locator('li[data-outcome="completed"]').count(), 5);
    for (const theme of ["dark", "light"]) {
      if (theme === "light") await page.getByRole("button", { name: "Switch to light theme" }).click();
      const colors = await progress.locator("li").evaluateAll(elements => {
        const luminance = (color) => color.match(/\d+/g).slice(0, 3)
          .map(Number).map(channel => channel / 255)
          .map(channel => channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4)
          .reduce((total, value, index) => total + value * [.2126, .7152, .0722][index], 0);
        return elements.map(el => {
          const color = getComputedStyle(el).color;
          const ink = luminance(color);
          const paper = luminance(getComputedStyle(el.closest(".composer-dock")).backgroundColor);
          return [color, (Math.max(ink, paper) + .05) / (Math.min(ink, paper) + .05)];
        });
      });
      assert.equal(new Set(colors.map(([color]) => color)).size, 6);
      assert.ok(colors.every(([, contrast]) => contrast >= 4.5));
      console.log(`Agent palette ${theme} contrast ratios: ${colors.map(([, ratio]) => ratio.toFixed(2)).join(", ")}`);
      await capture(page, `app-agent-palette-desktop-${theme}`);
    }

    assert.deepEqual(errors, []);
  } finally {
    await page.close();
  }

  const cachePage = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  try {
    await open(cachePage);
    await cachePage.getByRole("button", { name: /Quiz me on Diffusions/ }).click();
    await cachePage.getByRole("textbox", { name: "Ask about your notes" }).press("Control+Enter");
    await cachePage.getByText("Replaying a cached response", { exact: true }).waitFor();
    assert.equal(await cachePage.locator('.live-agents li[data-outcome="inactive"]').count(), 6);
    const cacheResponse = cachePage.locator(".assistant-message").filter({ hasText: "Cached practice question." });
    const cachePanel = cachePage.getByRole("complementary", { name: "Sources and concepts" });
    await cachePanel.getByText("Cached context").waitFor();
    await cachePanel.locator(".source-preview").filter({ hasText: /^Attention projections/ }).waitFor();
    const id = await cacheResponse.getAttribute("data-response-id");
    const cacheExecution = cacheResponse.getByRole("button", { name: "Execution details" });
    await cacheExecution.click();
    await cachePage.getByText("Replayed from cache. No agents ran for this turn.").waitFor();
    assert.equal(await cachePage.locator(".stage-duration").count(), 0);
    await cachePage.waitForFunction(() => !document.querySelector("#chat-input")?.readOnly);
    await reloadFromSnapshot(cachePage);
    await cachePanel.getByText("Cached context").waitFor();
    await cachePanel.locator(".source-preview").filter({ hasText: /^Attention projections/ }).waitFor();
    await cacheExecution.waitFor();
    assert.equal(await cacheResponse.getAttribute("data-response-id"), id);
  } finally {
    await cachePage.close();
  }
  if (reference && evidence) await captureStage3Reference(browser);
}

async function captureStage3Reference(browser) {
  for (const theme of ["dark", "light"]) {
    for (const [name, width, height] of [["desktop", 1440, 900], ["mobile", 390, 844]]) {
      const page = await browser.newPage({ viewport: { width, height } });
      try {
        await page.goto(`${reference}?theme=${theme}#study`, { waitUntil: "domcontentloaded" });
        const answer = page.locator("#answer-cache");
        if (name === "mobile") {
          await answer.locator("[data-response-sources]").click();
        } else {
          await answer.locator(".execution summary").click();
        }
        const sources = page.locator(name === "mobile" ? "#sources-dialog" : ".sources-panel");
        const note = sources.locator(".source-list details").first();
        await note.locator("summary").click();
        await waitForDisclosure(page, note, "p");
        if (name === "desktop") await waitForDisclosure(page, answer.locator(".execution"), ".execution-body");
        await page.evaluate(() => document.fonts.ready);
        await capture(page, `reference-local-${name}-${theme}`);
      } finally {
        await page.close();
      }
    }
  }
}

(async () => {
  const browser = await chromium.launch({
    args: ["--no-sandbox", "--no-zygote", "--renderer-process-limit=2"],
  });
  try {
    if (evidence) await fs.mkdir(evidence, { recursive: true });
    if (process.env.PRACTICE_TEST_MODE === "stage3") {
      await checkStage3(browser);
      console.log("Stage 3 sources and execution passed.");
      return;
    }
    if (process.env.PRACTICE_TEST_MODE === "history") {
      await checkStudyHistory(browser);
      console.log("Study history and layout passed.");
      return;
    }
    if (process.env.PRACTICE_TEST_MODE !== "layout") {
      await checkInteractions(browser);
      await checkStudyHistory(browser);
      console.log("Browser interactions passed.");
    }
    if (process.env.PRACTICE_TEST_MODE !== "interactions") {
      const matrix = await checkLayout(browser);
      console.log(JSON.stringify({ passed: matrix.length, matrix }, null, 2));
    }
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
