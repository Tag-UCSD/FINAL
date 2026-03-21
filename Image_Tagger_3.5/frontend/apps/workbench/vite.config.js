import { defineConfig } from 'vite';
import getBaseConfig from '../../vite.config.base';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
export default defineConfig(getBaseConfig(__dirname));