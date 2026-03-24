import React, { useState, useEffect, useMemo } from 'react';
import { Header, Button, ApiClient } from '@/lib';
import { Search, Filter, Download, Image as ImageIcon, CheckSquare, SlidersHorizontal, HelpCircle } from 'lucide-react';
import ImageDetailModal from './ImageDetailModal';

// Fallback demo data (used only if backend returns nothing)
const SAMPLE_IMAGES = Array.from({ length: 12 }).map((_, i) => ({
    id: i,
    url: `https://picsum.photos/seed/${i + 100}/400/${300 + (i % 3) * 50}`,
    tags: i % 2 === 0 ? ["Modern", "Kitchen", "High-Res"] : ["Traditional", "Living Room", "Low-Light"],
}));

const api = new ApiClient('/api/v1/explorer');

export default function ExplorerApp() {
    const [cart, setCart] = useState([]);
    const [debugMode, setDebugMode] = useState('none'); // 'none' | 'edges' | 'overlay' | 'depth' | 'complexity' | 'segmentation' | 'room' | 'materials' | 'materials2'
    const [segmentationConf, setSegmentationConf] = useState(0.25); // Segmentation confidence threshold
    const [overlayOpacity, setOverlayOpacity] = useState(0.5);
    const [edgeThresholds, setEdgeThresholds] = useState({ low: 50, high: 150 });
    const [query, setQuery] = useState("");
    const [filtersOpen, setFiltersOpen] = useState(true);

    const [images, setImages] = useState(SAMPLE_IMAGES);
    const [attributes, setAttributes] = useState([]);
    const [selectedTags, setSelectedTags] = useState([]);
    const [loading, setLoading] = useState(false);
    const [attrLoading, setAttrLoading] = useState(false);
    const [error, setError] = useState(null);
    const [seeding, setSeeding] = useState(false);
    const [seedError, setSeedError] = useState(null);
    const [seedMessage, setSeedMessage] = useState(null);
    const [seedElapsed, setSeedElapsed] = useState(0);
    const [seedSkipped, setSeedSkipped] = useState(false);
    const [detailIndex, setDetailIndex] = useState(null); // index into images[] for detail modal

    useEffect(() => {
        loadAttributes();
        // Initial search with empty filters/query
        runSearch([]);
    }, []);

    const groupedAttributes = useMemo(() => {
        if (!attributes || !attributes.length) return [];
        const groups = {};
        for (const attr of attributes) {
            const key = attr.key || "";
            const prefix = key.split('.')[0] || 'Other';
            if (!groups[prefix]) groups[prefix] = [];
            groups[prefix].push(attr);
        }
        return Object.entries(groups)
            .map(([name, items]) => ({
                name,
                items: items.slice(0, 8),
            }))
            .slice(0, 4);
    }, [attributes]);

    async function loadAttributes() {
        setAttrLoading(true);
        try {
            const data = await api.get('/attributes');
            if (Array.isArray(data)) {
                setAttributes(data);
            }
        } catch (err) {
            console.error('Failed to load attributes', err);
        } finally {
            setAttrLoading(false);
        }
    }

    async function runSearch(tagsOverride) {
        setLoading(true);
        setError(null);
        try {
            const filters = {};
            const tags = tagsOverride ?? selectedTags;
            if (tags && tags.length) {
                filters.tags = tags;
            }
            const data = await api.post('/search', {
                query_string: query,
                filters,
                page: 1,
                page_size: 48,
            });
            if (Array.isArray(data) && data.length) {
                setImages(data);
            } else {
                // If backend returns no results, keep UI consistent but empty
                setImages([]);
            }
        } catch (err) {
            console.error('Search failed', err);
            if (err && err.isMaintenance) {
                setError('__MAINTENANCE__:' + (err.message || 'System temporarily unavailable (503).'));
            } else {
                setError(err && err.message ? err.message : 'Search failed');
            }
        } finally {
            setLoading(false);
        }
    }

    async function handleSeed(force = false) {
        setSeedError(null);
        setSeedMessage(null);
        setSeedSkipped(false);
        const confirmed = window.confirm(
            force
                ? 'Force seed the sample dataset? This may create duplicates. Continue?'
                : 'Seed the bundled sample dataset (~9,497 entries)? This will add records to the database.'
        );
        if (!confirmed) return;
        setSeeding(true);
        setSeedElapsed(0);
        setSeedMessage('Seeding… this may take a few minutes.');
        try {
            const result = await api.post('/seed', { force });
            if (result && result.skipped) {
                setSeedMessage(result.message || 'Seeding skipped (already populated).');
                setSeedSkipped(true);
            } else {
                setSeedMessage(`Seeded ${result.created || 0} images.`);
                setSeedSkipped(false);
            }
            await runSearch();
        } catch (err) {
            setSeedError(err && err.message ? err.message : 'Seeding failed');
        } finally {
            setSeeding(false);
        }
    }

    useEffect(() => {
        if (!seeding) return;
        const timer = setInterval(() => {
            setSeedElapsed(prev => prev + 1);
        }, 1000);
        return () => clearInterval(timer);
    }, [seeding]);

    const handleQueryKeyDown = (e) => {
        if (e.key === 'Enter') {
            runSearch();
        }
    };

    const handleAttributeToggle = (key) => {
        setSelectedTags(prev => {
            const next = prev.includes(key)
                ? prev.filter(k => k !== key)
                : [...prev, key];
            // Run search with the next selection
            runSearch(next);
            return next;
        });
    };

    const toggleCart = (id) => {
        setCart(prev => (
            prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        ));
    };

    const selectAll = () => {
        setCart(images.map(img => img.id));
    };

    const handleExport = async () => {
        if (!cart.length) return;
        try {
            const payload = { image_ids: cart, format: 'json' };
            const data = await api.post('/export', payload);
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dataset_export.json';
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Export failed', err);
            if (err && err.isMaintenance) {
                setError('__MAINTENANCE__:' + (err.message || 'System temporarily unavailable (503).'));
            } else {
                setError('Export failed: ' + ((err && err.message) ? err.message : 'Unknown error'));
            }
        }
    };

    const isMaintenance = typeof error === 'string' && error.startsWith('__MAINTENANCE__:');
    const maintenanceMessage = isMaintenance ? error.replace('__MAINTENANCE__:', '') : null;

    return (
        <>
            <div className="relative flex flex-col h-screen bg-white">
                <Header appName="Explorer" title="Research Discovery" />

                {isMaintenance && (
                    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 text-gray-100">
                        <div className="bg-gray-900/95 px-6 py-4 rounded-lg shadow-lg max-w-md text-center">
                            <div className="font-semibold mb-2 text-sm">System temporarily unavailable</div>
                            <div className="text-xs mb-3">{maintenanceMessage || "The backend is reporting a maintenance / outage condition (503). Searches and exports are temporarily paused."}</div>
                            <div className="text-[10px] text-gray-400">If this persists for more than a few minutes, contact your TA or lab lead.</div>
                        </div>
                    </div>
                )}

                {/* Quick Help */}
                <div className="border-b border-blue-100 bg-blue-50 px-4 py-3 text-xs text-blue-900 flex gap-3 items-start">
                    <HelpCircle className="mt-0.5 flex-shrink-0" size={18} />
                    <div>
                        <div className="font-semibold text-sm mb-1">How to use Explorer</div>
                        <ol className="list-decimal ml-4 space-y-0.5">
                            <li>Start with a search query, or leave it blank to browse recent images.</li>
                            <li>Use <span className="font-semibold">Filters</span> to narrow by attributes, tags, or source.</li>
                            <li>Click any image to open the <span className="font-semibold">Detail View</span> — see full resolution, science attributes, debug modes, and human validations.</li>
                            <li>Use the <span className="font-semibold">checkbox button</span> (top-left of each card, visible on hover) to add images to your export cart.</li>
                            <li>Click <span className="font-semibold">Export Dataset</span> to download a JSON file for training or analysis.</li>
                        </ol>
                    </div>
                </div>

                {/* Toolbar */}
                <div className="border-b border-gray-200 p-4 flex gap-4 bg-gray-50 items-center z-20 shadow-sm">
                    <div className="relative flex-1 max-w-2xl">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                        <input
                            type="text"
                            placeholder='Query (e.g. "High prospect windows in modern style")'
                            className="w-full pl-10 pr-4 py-2.5 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none shadow-sm"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            onKeyDown={handleQueryKeyDown}
                        />
                    </div>
                    <Button variant="secondary" onClick={() => setFiltersOpen(!filtersOpen)}>
                        <SlidersHorizontal size={18} className="mr-2" /> Filters
                    </Button>
                    <Button onClick={() => runSearch()} disabled={loading}>
                        <Search size={18} className="mr-2" /> Search
                    </Button>
                    <Button
                        onClick={() => {
                            const next = debugMode === 'none'
                                ? 'edges'
                                : debugMode === 'edges'
                                    ? 'overlay'
                                    : debugMode === 'overlay'
                                        ? 'depth'
                                        : debugMode === 'depth'
                                            ? 'complexity'
                                            : debugMode === 'complexity'
                                                ? 'segmentation'
                                                : debugMode === 'segmentation'
                                                    ? 'room'
                                                    : debugMode === 'room'
                                                        ? 'materials'
                                                        : debugMode === 'materials'
                                                            ? 'materials2'
                                                            : 'none';
                            setDebugMode(next);
                        }}
                    >
                        <ImageIcon size={18} className="mr-2" />
                        {debugMode === 'none'
                            ? 'Debug: Off'
                            : debugMode === 'edges'
                                ? 'Debug: Edges'
                                : debugMode === 'overlay'
                                    ? 'Debug: Overlay'
                                    : debugMode === 'depth'
                                        ? 'Debug: Depth'
                                        : debugMode === 'complexity'
                                            ? 'Debug: Complexity'
                                            : debugMode === 'segmentation'
                                                ? 'Debug: Segmentation'
                                                : debugMode === 'room'
                                                    ? 'Debug: Room'
                                                    : debugMode === 'materials' ? 'Debug: Materials' : 'Debug: Materials2'}
                    </Button>
                    {(debugMode === 'edges' || debugMode === 'overlay' || debugMode === 'depth' || debugMode === 'complexity' || debugMode === 'segmentation' || debugMode === 'room' || debugMode === 'materials' || debugMode === 'materials2') && (
                        <div className="flex items-center gap-2 ml-3">
                            {(debugMode === 'edges' || debugMode === 'overlay' || debugMode === 'complexity') && (
                                <>
                                    <span className="text-xs text-gray-600 hidden md:inline">Edges</span>
                                    <input
                                        type="range"
                                        min={10}
                                        max={200}
                                        value={edgeThresholds.low}
                                        onChange={(e) =>
                                            setEdgeThresholds(prev => ({ ...prev, low: Number(e.target.value) }))
                                        }
                                    />
                                    <input
                                        type="range"
                                        min={50}
                                        max={300}
                                        value={edgeThresholds.high}
                                        onChange={(e) =>
                                            setEdgeThresholds(prev => ({ ...prev, high: Number(e.target.value) }))
                                        }
                                    />
                                </>
                            )}
                            {debugMode === 'overlay' && (
                                <>
                                    <span className="text-xs text-gray-600 hidden md:inline">Overlay</span>
                                    <input
                                        type="range"
                                        min={0}
                                        max={100}
                                        value={Math.round(overlayOpacity * 100)}
                                        onChange={(e) => setOverlayOpacity(Number(e.target.value) / 100)}
                                    />
                                </>
                            )}
                            {debugMode === 'depth' && (
                                <span className="text-xs text-gray-600 hidden md:inline">
                                    Depth debug uses model defaults (no sliders).
                                </span>
                            )}
                            {debugMode === 'complexity' && (
                                <span className="text-xs text-gray-600 hidden md:inline">
                                    Regionalized edge density heatmap
                                </span>
                            )}
                            {debugMode === 'segmentation' && (
                                <>
                                    <span className="text-xs text-black hidden md:inline">Confidence</span>
                                    <input
                                        type="range"
                                        min={10}
                                        max={90}
                                        value={Math.round(segmentationConf * 100)}
                                        onChange={(e) => setSegmentationConf(Number(e.target.value) / 100)}
                                        title={`Detection confidence: ${Math.round(segmentationConf * 100)}%`}
                                    />
                                    <span className="text-xs text-black w-8">{Math.round(segmentationConf * 100)}%</span>
                                    <span className="text-xs text-black hidden lg:inline ml-2">
                                        OneFormer-seg instance segmentation
                                    </span>
                                </>
                            )}
                            {debugMode === 'room' && (
                                <span className="text-xs text-black hidden md:inline">
                                    Places365 room type classification
                                </span>
                            )}
                            {debugMode === 'materials' && (
                                <span className="text-xs text-black hidden md:inline">
                                    Gemini Flash material detection (VLM)
                                </span>
                            )}
                            {debugMode === 'materials2' && (
                                <span className="text-xs text-black hidden md:inline">
                                    OneFormer + SigLIP2 material identification pipeline
                                </span>
                            )}
                        </div>
                    )}
                    <div className="flex-1"></div>

                    {/* Cart Summary */}
                    <div className="flex items-center gap-4 pl-6 border-l border-gray-300">
                        <div className="text-sm text-gray-600 hidden md:block">
                            <span className="font-bold text-gray-900 text-lg">{cart.length}</span> items
                        </div>
                        <Button onClick={handleExport} disabled={cart.length === 0}>
                            <Download size={18} className="mr-2" /> Export Dataset
                        </Button>
                    </div>
                </div>

                <div className="flex flex-1 overflow-hidden">
                    {/* Filters Sidebar */}
                    {filtersOpen && (
                        <aside className="w-72 border-r border-gray-200 bg-white overflow-y-auto flex-shrink-0 transition-all duration-300">
                            <div className="p-4 border-b border-gray-100 flex justify-between items-center">
                                <h3 className="font-bold text-gray-900 flex items-center gap-2">
                                    <Filter size={18} /> Refine
                                </h3>
                                <button onClick={() => setFiltersOpen(false)} className="text-gray-400 hover:text-gray-600 text-sm">
                                    Hide
                                </button>
                            </div>

                            <div className="p-4 space-y-8">
                                {attrLoading && (
                                    <p className="text-xs text-gray-500">Loading attribute taxonomy…</p>
                                )}

                                {!attrLoading && !attributes.length && (
                                    <p className="text-xs text-gray-500">
                                        Attribute registry is empty. Run install & seeding to load attributes.yml.
                                    </p>
                                )}

                                {groupedAttributes.map(group => (
                                    <div key={group.name}>
                                        <h4 className="font-semibold text-xs text-gray-500 uppercase tracking-wider mb-3">
                                            {group.name}
                                        </h4>
                                        <div className="space-y-2">
                                            {group.items.map(attr => {
                                                const checked = selectedTags.includes(attr.key);
                                                return (
                                                    <label
                                                        key={attr.id}
                                                        className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer hover:text-blue-600 group"
                                                    >
                                                        <input
                                                            type="checkbox"
                                                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                                            checked={checked}
                                                            onChange={() => handleAttributeToggle(attr.key)}
                                                        />
                                                        <span className="group-hover:translate-x-1 transition-transform">
                                                            <span title={attr.description || attr.name || attr.key}>{attr.name || attr.key}</span>
                                                        </span>
                                                    </label>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </aside>
                    )}

                    {/* Masonry Grid */}
                    <main className="flex-1 p-6 overflow-y-auto bg-gray-100">
                        <div className="flex justify-between items-center mb-4">
                            <p className="text-sm text-gray-500">
                                {loading
                                    ? "Loading results…"
                                    : `Showing ${images.length} result${images.length === 1 ? "" : "s"}`}
                            </p>
                            <button onClick={selectAll} className="text-sm text-blue-600 font-medium hover:underline">
                                Select All
                            </button>
                        </div>

                        {error && (
                            <div className="mb-4 text-xs text-red-600">
                                {error}
                            </div>
                        )}

                        {!error && !loading && images.length <= 1 && !query && (!selectedTags || !selectedTags.length) && (
                            <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-xs text-blue-900">
                                <div className="font-semibold text-sm mb-1">No data yet</div>
                                <div className="mb-2">
                                    Seed the bundled sample dataset to populate Explorer with example images.
                                </div>
                                <div className="flex items-center gap-3">
                                    <Button onClick={() => handleSeed(false)} disabled={seeding}>
                                        {seeding ? 'Seeding…' : 'Seed Sample Dataset'}
                                    </Button>
                                    {seedSkipped && (
                                        <Button variant="secondary" onClick={() => handleSeed(true)} disabled={seeding}>
                                            Force Seed
                                        </Button>
                                    )}
                                    {seeding && (
                                        <span className="text-xs text-blue-800">
                                            {seedMessage} ({seedElapsed}s)
                                        </span>
                                    )}
                                    {!seeding && seedMessage && (
                                        <span className="text-xs text-green-700">{seedMessage}</span>
                                    )}
                                    {seedError && <span className="text-xs text-red-600">{seedError}</span>}
                                </div>
                            </div>
                        )}

                        {(!images.length && !loading) && (
                            <div className="mt-10 flex flex-col items-center text-gray-400 text-sm">
                                <ImageIcon size={40} className="mb-3" />
                                <p>No results yet. Adjust your query or filters.</p>
                            </div>
                        )}

                        <div className="columns-1 sm:columns-2 md:columns-3 xl:columns-4 gap-6 space-y-6 pb-20">
                            {images.map((img, idx) => {
                                const tags = Array.isArray(img.tags) ? img.tags : [];
                                const inCart = cart.includes(img.id);
                                const altText = img.meta_data && img.meta_data.filename ? img.meta_data.filename : `Image ${img.id}`;
                                return (
                                    <div
                                        key={img.id}
                                        className={`break-inside-avoid group relative rounded-xl overflow-hidden border border-gray-200 shadow-sm cursor-pointer bg-white transition-all duration-200 ${inCart
                                            ? 'ring-2 ring-blue-500 transform scale-[1.02]'
                                            : 'hover:shadow-xl hover:-translate-y-1'
                                            }`}
                                        onClick={() => setDetailIndex(idx)}
                                        title="Click to inspect"
                                    >
                                        {/* Image */}
                                        {debugMode === 'overlay' ? (
                                            <div className="relative w-full">
                                                <img
                                                    src={img.url}
                                                    alt={altText}
                                                    className="w-full h-auto block"
                                                    loading="lazy"
                                                />
                                                <img
                                                    src={`/api/v1/debug/images/${img.id}/edges?t1=${edgeThresholds.low}&t2=${edgeThresholds.high}`}
                                                    alt={`Edges for image ${img.id}`}
                                                    className="w-full h-auto block absolute inset-0 pointer-events-none mix-blend-screen"
                                                    style={{ opacity: overlayOpacity }}
                                                />
                                            </div>
                                        ) : (
                                            <img
                                                src={debugMode === 'edges'
                                                    ? `/api/v1/debug/images/${img.id}/edges?t1=${edgeThresholds.low}&t2=${edgeThresholds.high}`
                                                    : debugMode === 'depth'
                                                        ? `/api/v1/debug/images/${img.id}/depth`
                                                        : debugMode === 'complexity'
                                                            ? `/api/v1/debug/images/${img.id}/complexity?t1=${edgeThresholds.low}&t2=${edgeThresholds.high}`
                                                            : debugMode === 'segmentation'
                                                                ? `/api/v1/debug/images/${img.id}/segmentation?conf=${segmentationConf}`
                                                                : debugMode === 'room'
                                                                    ? `/api/v1/debug/images/${img.id}/room`
                                                                    : debugMode === 'materials'
                                                                        ? `/api/v1/debug/images/${img.id}/materials`
                                                                        : debugMode === 'materials2'
                                                                            ? `/api/v1/debug/images/${img.id}/materials2`
                                                                            : img.url}
                                                alt={altText}
                                                className="w-full h-auto block"
                                                loading="lazy"
                                            />
                                        )}

                                        {/* Tag count badge */}
                                        <div className="absolute top-2 right-2 bg-black/60 backdrop-blur text-white text-xs font-semibold px-2 py-1 rounded">
                                            {tags.length} tags
                                        </div>

                                        {/* Explicit cart checkbox — stopPropagation so it doesn't open the modal */}
                                        <button
                                            className={`absolute top-2 left-2 flex items-center justify-center w-7 h-7 rounded-full shadow-lg border-2 transition-colors ${inCart
                                                ? 'bg-blue-600 border-blue-600 text-white'
                                                : 'bg-white/80 border-gray-300 text-gray-400 opacity-0 group-hover:opacity-100'
                                                }`}
                                            onClick={e => { e.stopPropagation(); toggleCart(img.id); }}
                                            title={inCart ? 'Remove from cart' : 'Add to cart'}
                                            aria-label={inCart ? 'Remove from cart' : 'Add to cart'}
                                        >
                                            <CheckSquare size={14} />
                                        </button>

                                        {/* Cart ring highlight */}
                                        {inCart && (
                                            <div className="absolute inset-0 ring-inset ring-2 ring-blue-500 pointer-events-none rounded-xl" />
                                        )}

                                        {/* Tags */}
                                        <div className="p-3">
                                            <div className="flex flex-wrap gap-1">
                                                {tags.map(t => (
                                                    <span
                                                        key={t}
                                                        className="bg-gray-100 text-gray-600 text-[10px] uppercase font-bold px-2 py-1 rounded"
                                                    >
                                                        {t}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </main>
                </div>
            </div>

            {/* Image detail modal */}
            {detailIndex !== null && images.length > 0 && (
                <ImageDetailModal
                    images={images}
                    initialIndex={detailIndex}
                    debugMode={debugMode}
                    edgeThresholds={edgeThresholds}
                    overlayOpacity={overlayOpacity}
                    segmentationConf={segmentationConf}
                    onClose={() => setDetailIndex(null)}
                    onAddToCart={toggleCart}
                    cart={cart}
                />
            )}
        </>
    );
}