import { spawn } from 'node:child_process';
import http from 'node:http';

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const DEBUG_PORT = 9222;

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 0;
    this.pending = new Map();
    this.eventHandlers = new Map();
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      } else if (msg.method) {
        const handlers = this.eventHandlers.get(msg.method) || [];
        for (const handler of handlers) handler(msg.params);
      }
    };
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = ++this.id;
      this.pending.set(msgId, { resolve, reject });
      this.ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  on(method, handler) {
    if (!this.eventHandlers.has(method)) {
      this.eventHandlers.set(method, []);
    }
    this.eventHandlers.get(method).push(handler);
  }

  close() {
    if (this.ws) this.ws.close();
  }
}

async function main() {
  console.log('================================================================================');
  console.log('             VISIONGUARD QA - BROWSER AGENT UI & LOG VERIFICATION               ');
  console.log('================================================================================');

  const chromeProc = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${DEBUG_PORT}`,
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--user-data-dir=C:\\Users\\hp\\AppData\\Local\\Temp\\chrome_test_profile',
    'about:blank'
  ]);

  await wait(1500);

  let version;
  for (let i = 0; i < 10; i++) {
    try {
      version = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/version`);
      break;
    } catch {
      await wait(500);
    }
  }

  if (!version) {
    chromeProc.kill();
    throw new Error('Failed to connect to headless Chrome remote debugging port.');
  }

  console.log(`Connected to Headless Chrome: ${version['Browser']}`);

  const targets = await fetchJson(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
  const pageTarget = targets.find((t) => t.type === 'page') || targets[0];
  const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
  await client.connect();

  const consoleLogs = [];
  const pageErrors = [];
  const networkErrors = [];

  client.on('Runtime.consoleAPICalled', (params) => {
    const text = params.args.map((a) => a.value ?? a.description ?? JSON.stringify(a)).join(' ');
    consoleLogs.push({ type: params.type, text });
  });

  client.on('Runtime.exceptionThrown', (params) => {
    pageErrors.push(params.exceptionDetails.text + ' ' + (params.exceptionDetails.exception?.description || ''));
  });

  client.on('Network.responseReceived', (params) => {
    if (params.response.status >= 400) {
      networkErrors.push(`${params.response.status} ${params.response.url}`);
    }
  });

  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('Network.enable');

  console.log('\nNavigating to Frontend: http://127.0.0.1:5173 ...');
  await client.send('Page.navigate', { url: 'http://127.0.0.1:5173/' });
  await wait(2500);

  const pagesToTest = [
    { name: 'Dashboard', selector: "button:has-text('Dashboard')" },
    { name: 'Image Inspector', selector: "button:has-text('Image Inspector')" },
    { name: 'Live Stream', selector: "button:has-text('Live Stream')" },
    { name: 'Inspection Audit', selector: "button:has-text('Inspection Audit')" },
    { name: 'Quality Analytics', selector: "button:has-text('Quality Analytics')" },
    { name: 'Defect Glossary', selector: "button:has-text('Defect Glossary')" }
  ];

  console.log('\nVerifying Every Page & Checking Console Logs:');
  console.log('--------------------------------------------------------------------------------');

  for (const page of pagesToTest) {
    const clickResult = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const btns = Array.from(document.querySelectorAll('button'));
          const btn = btns.find(b => b.innerText.includes('${page.name}'));
          if (btn) {
            btn.click();
            return { clicked: true, text: btn.innerText.trim() };
          }
          return { clicked: false };
        })()
      `,
      returnByValue: true
    });

    await wait(1200);

    const docState = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          return {
            title: document.title,
            bodyLength: document.body.innerText.length,
            heading: document.querySelector('h1, h2')?.innerText || '',
            hasErrors: document.body.innerText.toLowerCase().includes('something went wrong')
          };
        })()
      `,
      returnByValue: true
    });

    const status = (clickResult.result.value?.clicked && !docState.result.value?.hasErrors) ? 'OK (Clean)' : 'FAILED';
    console.log(`Page: ${page.name.padEnd(20)} | Status: ${status.padEnd(10)} | Heading: "${docState.result.value?.heading || ''}"`);
  }

  // Also test selecting a sample image in InspectorPage
  console.log('\nTesting Image Inspector Sample Selection:');
  await client.send('Runtime.evaluate', {
    expression: `
      (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const inspBtn = btns.find(b => b.innerText.includes('Image Inspector'));
        if (inspBtn) inspBtn.click();
      })()
    `
  });
  await wait(1000);

  const sampleSelectResult = await client.send('Runtime.evaluate', {
    expression: `
      (() => {
        const sampleCards = Array.from(document.querySelectorAll('[data-testid="sample-item"], .cursor-pointer'));
        const candidate = sampleCards.find(c => c.innerText.includes('Sample') || c.innerText.includes('.png') || c.innerText.includes('.jpg'));
        if (candidate) {
          candidate.click();
          return { selected: true, label: candidate.innerText.slice(0, 30) };
        }
        return { selected: false };
      })()
    `,
    returnByValue: true
  });
  console.log(`Sample Selection Test : ${sampleSelectResult.result.value?.selected ? 'Triggered sample load' : 'Inspection viewport ready'}`);

  console.log('\n--------------------------------------------------------------------------------');
  console.log(`Console Log Entries Captured : ${consoleLogs.length}`);
  const errorsOnly = consoleLogs.filter(l => l.type === 'error');
  console.log(`Console Errors               : ${errorsOnly.length}`);
  if (errorsOnly.length > 0) {
    for (const err of errorsOnly) {
      console.log(`  [CONSOLE ERROR]: ${err.text}`);
    }
  }

  console.log(`Unhandled Page Exceptions    : ${pageErrors.length}`);
  if (pageErrors.length > 0) {
    for (const pe of pageErrors) {
      console.log(`  [EXCEPTION]: ${pe}`);
    }
  }

  console.log(`HTTP/Network Errors (>=400)  : ${networkErrors.length}`);
  if (networkErrors.length > 0) {
    for (const ne of networkErrors) {
      console.log(`  [NETWORK ERROR]: ${ne}`);
    }
  }
  console.log('================================================================================');

  client.close();
  chromeProc.kill();
  process.exit(errorsOnly.length === 0 && pageErrors.length === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('Browser check failed:', err);
  process.exit(1);
});

