// A tiny static file server (node:http only) for the web build, used by run_web_checks.mjs.
// Serves the right MIME types for .wasm/.mjs and refuses paths outside the root.
//   node tools/web/serve.mjs [root=build-web/dist] [port=8080]   (manual use)
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.png': 'image/png',
  '.ply': 'application/octet-stream',
  '.gz': 'application/gzip',
  '.data': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
};

// Starts serving `root` on 127.0.0.1:`port` (0 = any free port). Resolves to {url, close}.
// `mounts` maps URL prefixes to extra directories, e.g. {'/__testdata/': 'data'}: the automated
// tests serve the local scenes this way. They are never copied into the site (dist/), which is
// what gets deployed; the INRIA scenes in data/scenes must never be published.
export function startServer(root, port = 0, mounts = {}) {
  const absRoot = path.resolve(root);
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    let base = absRoot;
    let rel = urlPath;
    for (const [prefix, dir] of Object.entries(mounts)) {
      if (urlPath.startsWith(prefix)) {
        base = path.resolve(dir);
        rel = urlPath.slice(prefix.length - 1);
        break;
      }
    }
    let file = path.join(base, rel);
    if (!file.startsWith(base)) {
      res.writeHead(403).end();
      return;
    }
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) {
        res.writeHead(404, { 'Content-Type': 'text/plain' }).end(`not found: ${urlPath}`);
        return;
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
        'Content-Length': st.size,
        'Cache-Control': 'no-store',
      });
      fs.createReadStream(file).pipe(res);  // streamed: scene files can be hundreds of MB
    });
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () => {
      const { port: actual } = server.address();
      resolve({ url: `http://127.0.0.1:${actual}/`, close: () => new Promise((r) => server.close(r)) });
    });
  });
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const root = process.argv[2] || path.join(here, '..', '..', 'build-web', 'dist');
  const { url } = await startServer(root, parseInt(process.argv[3] || '8080', 10));
  console.log(`Serving ${path.resolve(root)} at ${url}`);
}
