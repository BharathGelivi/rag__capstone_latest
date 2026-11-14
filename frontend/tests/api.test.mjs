import { test } from 'node:test';
import assert from 'node:assert/strict';
import { streamChat } from '../src/lib/api.ts';

test('chat stream rejects HTTP failure and incomplete streams, preserves split UTF-8 frames', async () => {
  const original = globalThis.fetch;
  const consume = async () => { const events = []; for await (const event of streamChat({})) events.push(event); return events; };
  try {
    globalThis.fetch = async () => new Response('Internal details', { status: 500 });
    await assert.rejects(consume(), /HTTP 500/);
    globalThis.fetch = async () => new Response('data: {"event":"token","text":"partial"}\n\n');
    await assert.rejects(consume(), /before completion/);
    const bytes = new TextEncoder().encode('data: {"event":"token","text":"Evidence → claim"}\n\ndata: {"event":"done"}\n\n');
    let cancelled = false;
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) { for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); },
      cancel() { cancelled = true; },
    }));
    // Early exit also releases the reader and underlying connection.
    for await (const event of streamChat({})) {
      assert.equal(event.text, 'Evidence → claim');
      break;
    }
    assert.equal(cancelled, true);
    globalThis.fetch = async () => new Response(bytes);
    assert.deepEqual((await consume()).map(e => e.event), ['token', 'done']);
    const crlf = new TextEncoder().encode('data: {"event":"done"}\r\n\r\n');
    globalThis.fetch = async () => new Response(new ReadableStream({
      start(controller) {
        for (const byte of crlf) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      },
    }));
    assert.deepEqual((await consume()).map(e => e.event), ['done']);
  } finally {
    globalThis.fetch = original;
  }
});
