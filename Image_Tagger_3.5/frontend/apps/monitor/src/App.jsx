import React, { useEffect, useState, useMemo } from 'react';
import { Header, ApiClient, Button } from '@shared';
import { AlertTriangle, TrendingUp, Users, Activity, BarChart2, RefreshCcw, Eye, X } from 'lucide-react';

const api = new ApiClient('/api/v1/monitor', { 'X-User-Role': 'admin' });
const debugApi = new ApiClient('/api/v1/debug', { 'X-User-Role': 'admin' });

export default function MonitorApp() {
    const [teamStats, setTeamStats] = useState([]);
    const [irrStats, setIrrStats] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [inspectorImage, setInspectorImage] = useState(null);
    const [inspectorRecords, setInspectorRecords] = useState([]);
    const [inspectorLoading, setInspectorLoading] = useState(false);
    const [inspectorError, setInspectorError] = useState(null);
    const [pipelineHealth, setPipelineHealth] = useState(null);
    const [pipelineLoading, setPipelineLoading] = useState(false);
    const [pipelineError, setPipelineError] = useState(null);

    useEffect(() => {
        loadData();
    }, []);

    async function loadData() {
        setLoading(true);
        setError(null);
        try {
            const [velocity, irr, pipeline] = await Promise.all([
                api.get('/velocity'),
                api.get('/irr'),
                debugApi.get('/pipeline_health'),
            ]);
            setTeamStats(Array.isArray(velocity) ? velocity : []);
            setIrrStats(Array.isArray(irr) ? irr : []);
            setPipelineHealth(pipeline || null);
        } catch (err) {
            console.error('Failed to load supervisor data', err);
            setError(err.message || 'Failed to load supervisor dashboard');
        } finally {
            setLoading(false);
            setPipelineLoading(false);
        }
    }

    const totalValidations = useMemo(
        () => teamStats.reduce((sum, u) => sum + (u.images_validated || 0), 0),
        [teamStats]
    );

    const flaggedCount = useMemo(
        () => teamStats.filter(u => u.status === 'flagged').length,
        [teamStats]
    );

    const globalIRR = useMemo(() => {
        if (!irrStats.length) return null;
        const total = irrStats.reduce((sum, r) => sum + (r.agreement_score || 0), 0);
        return total / irrStats.length;
    }, [irrStats]);

    const activeTaggers = useMemo(
        () => teamStats.length,
        [teamStats]
    );


async function loadPipelineHealth() {
    setPipelineLoading(true);
    setPipelineError(null);
    try {
        const data = await debugApi.get('/pipeline_health');
        setPipelineHealth(data || null);
    } catch (err) {
        console.error('Failed to load pipeline health', err);
        setPipelineError(err && err.message ? err.message : 'Failed to load pipeline health');
    } finally {
        setPipelineLoading(false);
    }
}
    async function openInspector(row) {
        setInspectorImage({ image_id: row.image_id, filename: row.filename });
        setInspectorRecords([]);
        setInspectorError(null);
        setInspectorLoading(true);
        try {
            const data = await api.get(`/image/${row.image_id}/validations`);
            setInspectorRecords(Array.isArray(data) ? data : []);
        } catch (err) {
            console.error('Failed to load inspection data', err);
            setInspectorError(err.message || 'Failed to load inspection data');
        } finally {
            setInspectorLoading(false);
        }
    }

    function closeInspector() {
        setInspectorImage(null);
        setInspectorRecords([]);
        setInspectorError(null);
    }

    return (
        <div className="min-h-screen bg-gray-100 pb-10">
            <Header appName="Supervisor" title="Quality Control Dashboard" />

            <div className="p-8 max-w-7xl mx-auto space-y-8">
                {/* Controls / Status */}
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <p className="text-sm text-gray-500">
                            Monitor team velocity, agreement, and potential quality issues.
                        </p>
                        {error && (
                            <p className="text-xs text-red-600 mt-1">
                                {error}
                            </p>
                        )}
                    </div>
                    <div className="flex items-center gap-3">
                        {loading && (
                            <span className="text-xs text-gray-500 flex items-center gap-2">
                                <Activity className="animate-spin" size={14} /> Updating…
                            </span>
                        )}
                        <Button variant="secondary" onClick={loadData}>
                            <RefreshCcw size={16} className="mr-2" /> Refresh
                        </Button>
                    </div>
                </div>


<div className="mt-2 p-2 rounded-md bg-blue-50 border border-blue-100 text-[11px] text-blue-900 space-y-1">
    <p className="font-semibold">How to use the Supervisor dashboard</p>
    <ul className="list-disc ml-4 space-y-0.5">
        <li>This view is for supervisors and PIs to track team throughput, agreement, and possible quality issues.</li>
        <li>Use the metrics above to spot low volume, low IRR, or spikes in errors that may need investigation.</li>
        <li>Drill into specific images or annotators using Tag Inspector when a pattern looks suspicious.</li>
        <li>If metric tiles fail to load or look wrong, treat that as a system issue and notify an engineer.</li>
    </ul>
</div>

                {/* Top Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <MetricCard
                        icon={<Activity />}
                        label="Total Validations"
                        value={totalValidations}
                        unit="decisions logged"
                    />
                    <MetricCard
                        icon={<TrendingUp />}
                        label="Global IRR"
                        value={globalIRR !== null ? globalIRR.toFixed(2) : '—'}
                        unit="agreement"
                    />
                    <MetricCard
                        icon={<Users />}
                        label="Active Taggers"
                        value={activeTaggers}
                        unit="users"
                    />
                    <MetricCard
                        icon={<AlertTriangle />}
                        label="Flagged Taggers"
                        value={flaggedCount}
                        unit="needs review"
                        danger={flaggedCount > 0}
                    />
                </div>


{/* Science Pipeline Health */}
<section className="mt-4">
    <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
            <BarChart2 size={16} className="text-blue-600" />
            <h2 className="text-sm font-semibold text-gray-900">
                Science pipeline health
            </h2>
        </div>
        <div className="flex items-center gap-2">
            {pipelineHealth && (
                <span className="text-[11px] text-gray-500">
                    Last check: contracts by tier for configured analyzers.
                </span>
            )}
            <Button
                variant="ghost"
                size="xs"
                onClick={loadPipelineHealth}
                disabled={pipelineLoading}
            >
                <RefreshCcw size={14} className="mr-1" />
                {pipelineLoading ? 'Checking…' : 'Re-check'}
            </Button>
        </div>
    </div>

    <div className="p-3 rounded-lg border border-gray-200 bg-white text-xs text-gray-800 space-y-2">
        <p className="text-[11px] text-blue-900">
            This check instantiates each analyzer once and reports its tier and contracts.
            It does <span className="font-semibold">not</span> process real images, but it will
            catch most import/contract breakage before a full science run fails.
        </p>

        <div className="flex flex-wrap gap-4 items-center">
            <div>
                <p className="text-[10px] uppercase tracking-wide text-gray-500 font-semibold">
                    Import status
                </p>
                <p className="text-sm font-medium">
                    {pipelineHealth
                        ? (pipelineHealth.import_ok ? 'OK' : 'FAILED')
                        : 'Not checked yet'}
                </p>
            </div>
            <div>
                <p className="text-[10px] uppercase tracking-wide text-gray-500 font-semibold">
                    OpenCV
                </p>
                <p className="text-sm font-medium">
                    {pipelineHealth
                        ? (pipelineHealth.cv2_available ? 'available' : 'missing')
                        : '—'}
                </p>
            </div>
            {pipelineError && (
                <div className="flex-1 text-[11px] text-red-600">
                    {pipelineError}
                </div>
            )}
        </div>

        {pipelineHealth && pipelineHealth.analyzers_by_tier && (
            <div className="mt-2">
                <p className="text-[11px] font-semibold text-gray-700 mb-1">
                    Analyzers by tier
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {Object.entries(pipelineHealth.analyzers_by_tier).map(([tier, analyzers]) => (
                        <div key={tier} className="border border-gray-100 rounded-md p-2 bg-gray-50">
                            <p className="text-[11px] font-semibold text-gray-800 mb-1">
                                Tier {tier} ({Array.isArray(analyzers) ? analyzers.length : 0})
                            </p>
                            <ul className="space-y-0.5">
                                {Array.isArray(analyzers) && analyzers.map((a) => (
                                    <li key={a.name} className="flex flex-col">
                                        <span className="text-[11px] font-medium text-gray-900">
                                            {a.name}
                                        </span>
                                        <span className="text-[10px] text-gray-500">
                                            requires: {Array.isArray(a.requires) && a.requires.length
                                                ? a.requires.join(', ')
                                                : '—'}
                                        </span>
                                        <span className="text-[10px] text-gray-500">
                                            provides: {Array.isArray(a.provides) && a.provides.length
                                                ? a.provides.join(', ')
                                                : '—'}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </div>
            </div>
        )}

        {pipelineHealth && pipelineHealth.warnings && pipelineHealth.warnings.length > 0 && (
            <div className="mt-2">
                <p className="text-[11px] font-semibold text-amber-700 mb-1">
                    Warnings
                </p>
                <ul className="list-disc ml-4 space-y-0.5 text-[11px] text-amber-800">
                    {pipelineHealth.warnings.map((w, idx) => (
                        <li key={idx}>{w}</li>
                    ))}
                </ul>
            </div>
        )}

        {pipelineHealth && pipelineHealth.analyzer_errors && pipelineHealth.analyzer_errors.length > 0 && (
            <div className="mt-2">
                <p className="text-[11px] font-semibold text-red-700 mb-1">
                    Analyzer errors
                </p>
                <ul className="list-disc ml-4 space-y-0.5 text-[11px] text-red-800">
                    {pipelineHealth.analyzer_errors.map((e, idx) => (
                        <li key={idx}>
                            {e.analyzer}: {e.error}
                        </li>
                    ))}
                </ul>
            </div>
        )}
    </div>
</section>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Team Velocity Table */}
                    <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                        <div className="p-4 flex items-center justify-between border-b border-gray-100">
                            <div className="flex items-center gap-2">
                                <BarChart2 className="text-blue-500" size={18} />
                                <h2 className="font-semibold text-gray-900 text-sm">
                                    Tagger Velocity & Quality
                                </h2>
                            </div>
                            <span className="text-xs text-gray-400">
                                Based on Validation.duration_ms and per-user totals
                            </span>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead>
                                    <tr className="text-xs font-bold text-gray-500 uppercase tracking-wider border-b">
                                        <th className="pb-3 pl-2">User</th>
                                        <th className="pb-3">Validations</th>
                                        <th className="pb-3">Avg. Dwell</th>
                                        <th className="pb-3 text-right pr-2">Status</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                    {teamStats.map(user => (
                                        <TaggerRow key={user.user_id} user={user} />
                                    ))}
                                    {!teamStats.length && !loading && (
                                        <tr>
                                            <td colSpan={4} className="py-6 text-center text-xs text-gray-400">
                                                No validations found yet. Run a tagging session in Workbench.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* IRR / Agreement Heatmap */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                        <div className="p-4 flex items-center justify-between border-b border-gray-100">
                            <div className="flex items-center gap-2">
                                <BarChart2 className="text-purple-500" size={18} />
                                <h2 className="font-semibold text-gray-900 text-sm">
                                    Inter-Rater Agreement
                                </h2>
                            </div>
                            <span className="text-xs text-gray-400">
                                Simple majority-based agreement per image
                            </span>
                        </div>

                        <div className="p-4">
                            <p className="text-xs text-gray-500 mb-3">
                                Rows show images with multiple raters. Agreement is the proportion
                                of raters that match the majority decision (value &gt; 0.5 vs ≤ 0.5).
                            </p>

                            <div className="overflow-x-auto">
                                <table className="w-full text-left">
                                    <thead>
                                        <tr className="text-xs font-bold text-gray-500 uppercase tracking-wider border-b">
                                            <th className="pb-3">Image</th>
                                            <th className="pb-3">Agreement</th>
                                            <th className="pb-3">Conflicts</th>
                                            <th className="pb-3">Raters</th>
                                            <th className="pb-3 text-right">Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-100">
                                        {irrStats.map(row => (
                                            <IRRRow key={row.image_id} row={row} onInspect={() => openInspector(row)} />
                                        ))}
                                        {!irrStats.length && !loading && (
                                            <tr>
                                                <td colSpan={4} className="py-6 text-center text-xs text-gray-400">
                                                    No overlapping ratings yet. Encourage multiple taggers to rate the same images.
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                {inspectorImage && (
                    <InspectorDrawer
                        image={inspectorImage}
                        records={inspectorRecords}
                        loading={inspectorLoading}
                        error={inspectorError}
                        onClose={closeInspector}
                    />
                )}
            </div>
        </div>
    );
}

function MetricCard({ icon, label, value, unit, danger = false }) {
    return (
        <div className={`bg-white p-6 rounded-xl shadow border-l-4 ${
            danger ? 'border-red-500' : 'border-blue-500'
        } flex items-center justify-between`}>
            <div>
                <p className="text-gray-500 text-xs font-bold uppercase tracking-wider">{label}</p>
                <h3 className="text-2xl font-bold text-gray-900 mt-1">{value}</h3>
                <p className="text-xs text-gray-400 font-medium">{unit}</p>
            </div>
            <div className={`p-3 rounded-lg ${danger ? 'bg-red-50 text-red-500' : 'bg-blue-50 text-blue-600'}`}>
                {icon}
            </div>
        </div>
    );
}

function TaggerRow({ user }) {
    const avgSec = (user.avg_duration_ms || 0) / 1000.0;
    const isFlagged = user.status === 'flagged';

    return (
        <tr className={`hover:bg-gray-50 transition-colors ${isFlagged ? 'bg-red-50/40' : ''}`}>
            <td className="py-3 pl-2 font-medium text-gray-900">
                {user.username || `user-${user.user_id}`}
            </td>
            <td className="py-3 text-gray-600">
                {user.images_validated} validations
            </td>
            <td className="py-3 text-gray-600">
                {avgSec.toFixed(2)}s / decision
            </td>
            <td className="py-3 pr-2 text-right">
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-[11px] font-semibold ${
                    isFlagged
                        ? 'bg-red-100 text-red-700'
                        : 'bg-emerald-100 text-emerald-700'
                }`}>
                    {isFlagged ? 'Flagged' : 'Healthy'}
                </span>
            </td>
        </tr>
    );
}

function IRRRow({ row, onInspect }) {
    const agreePct = (row.agreement_score || 0) * 100;
    const conflict = row.conflict_count || 0;
    const raters = Array.isArray(row.raters) ? row.raters : [];

    const severity =
        agreePct >= 90 ? 'high' :
        agreePct >= 70 ? 'medium' :
        'low';

    let badgeClass = 'bg-emerald-100 text-emerald-700';
    if (severity === 'medium') {
        badgeClass = 'bg-amber-100 text-amber-700';
    } else if (severity === 'low') {
        badgeClass = 'bg-red-100 text-red-700';
    }

    return (
        <tr className="hover:bg-gray-50 transition-colors">
            <td className="py-3 text-gray-900 text-sm">
                {row.filename}
            </td>
            <td className="py-3">
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-[11px] font-semibold ${badgeClass}`}>
                    {agreePct.toFixed(0)}% agreement
                </span>
            </td>
            <td className="py-3 text-gray-600 text-sm">
                {conflict} conflict{conflict === 1 ? '' : 's'}
            </td>
            <td className="py-3 text-right text-xs text-gray-500">
                {raters.join(', ')}
            </td>
        </tr>
    );
}


function InspectorDrawer({ image, records, loading, error, onClose }) {
    const [detail, setDetail] = useState(null);
    const [detailLoading, setDetailLoading] = useState(true);
    const [detailError, setDetailError] = useState(null);
    const [showHelp, setShowHelp] = useState(false);

    useEffect(() => {
        if (!image || !image.image_id) {
            return;
        }

        let cancelled = false;

        async function fetchInspector() {
            setDetailLoading(true);
            setDetailError(null);
            try {
                const resp = await fetch(`/api/v1/monitor/image/${image.image_id}/inspector`);
                if (!resp.ok) {
                    throw new Error(`Inspector HTTP ${resp.status}`);
                }
                const data = await resp.json();
                if (!cancelled) {
                    setDetail(data);
                }
            } catch (err) {
                console.error('Failed to load inspector detail', err);
                if (!cancelled) {
                    setDetailError(err.message || 'Failed to load inspector detail');
                }
            } finally {
                if (!cancelled) {
                    setDetailLoading(false);
                }
            }
        }

        fetchInspector();

        return () => {
            cancelled = true;
        };
    }, [image?.image_id]);

    const isBusy = loading || detailLoading;
    const effectiveError = error || detailError;

    const tags = detail?.tags || [];
    const features = detail?.features || [];
    const inspectorValidations = detail?.validations || records || [];

    return (
        <div className="fixed inset-0 bg-black/30 flex justify-end z-40">
            <div className="w-full max-w-5xl bg-white h-full shadow-xl flex flex-col">
                <div className="px-4 py-3 border-b flex items-center justify-between">
                    <div>
                        <h3 className="text-sm font-semibold text-gray-900">
                            Tag Inspector
                        </h3>
                        <p className="text-xs text-gray-500">
                            Image {image.image_id} · {image.filename}
                        </p>
                        {detail?.image?.url && (
                            <p className="text-[11px] text-gray-400">
                                Source: {detail.image.url}
                            </p>
                        )}
                    </div>

<div className="flex items-center gap-1">
    <button
        type="button"
        onClick={() => setShowHelp(prev => !prev)}
        className="p-1 rounded-full hover:bg-gray-100 text-gray-500"
    >
        <HelpCircle size={16} />
    </button>
    <button
        type="button"
        onClick={onClose}
        className="p-1 rounded-full hover:bg-gray-100 text-gray-500"
    >
        <X size={16} />
    </button>
</div>
                </div>


{showHelp && (
    <div className="mx-4 my-2 p-2 rounded-md bg-blue-50 border border-blue-100 text-[11px] text-blue-900 space-y-1">
        <p className="font-semibold">How to read Tag Inspector</p>
        <ul className="list-disc ml-4 space-y-0.5">
            <li>This view is read-only and for supervisors/admins.</li>
            <li>The top snapshot shows how many science attributes, indices, and the IRR value (if available) were computed for this image.</li>
            <li>The High-level indices table lists the composite indices that can feed BN models (e.g., restorativeness, clutter, fluency).</li>
            <li>The Science attributes table shows the raw numeric features computed by the pipeline for this particular room image.</li>
            <li>The Human validations table shows who rated which attributes, with values, dwell times, and timestamps.</li>
        </ul>
    </div>
)}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
                    {isBusy && (
                        <div className="text-gray-500 flex items-center gap-2">
                            <Activity className="animate-spin" size={14} />
                            Loading inspector data…
                        </div>
                    )}

                    {effectiveError && (
                        <div className="text-red-600 text-xs">
                            {effectiveError}
                        </div>
                    )}

                    {!isBusy && !effectiveError && (
                        <>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
                                {/* Image preview + basic meta */}
                                <div className="md:col-span-1 space-y-3">
                                    <div className="border rounded-lg overflow-hidden bg-gray-50">
                                        {detail?.image?.url ? (
                                            <img
                                                src={detail.image.url}
                                                alt={image.filename}
                                                className="w-full h-40 object-contain bg-black/5"
                                            />
                                        ) : (
                                            <div className="h-40 flex items-center justify-center text-gray-400 text-[11px]">
                                                No preview available
                                            </div>
                                        )}
                                    </div>
                                    <div className="space-y-1">
                                        <div className="font-semibold text-[11px] text-gray-700">
                                            Science snapshot
                                        </div>
                                        <div className="text-[11px] text-gray-500 space-y-0.5">
                                            <div>
                                                <span className="font-medium">Science attrs:</span>{" "}
                                                {features.length}
                                            </div>
                                            <div>
                                                <span className="font-medium">Indices:</span>{" "}
                                                {tags.length}
                                            </div>
                                            {detail?.bn?.irr !== undefined && detail.bn.irr !== null && (
                                                <div>
                                                    <span className="font-medium">IRR:</span>{" "}
                                                    {detail.bn.irr.toFixed
                                                        ? detail.bn.irr.toFixed(3)
                                                        : detail.bn.irr}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* High-level indices / tags */}
                                <div className="md:col-span-2">
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="font-semibold text-[11px] text-gray-700">
                                            High-level indices (BN inputs)
                                        </div>
                                        <div className="text-[11px] text-gray-400">
                                            {tags.length ? `${tags.length} indices` : "No indices yet"}
                                        </div>
                                    </div>
                                    <div className="border rounded-lg max-h-56 overflow-y-auto">
                                        {tags.length ? (
                                            <table className="w-full text-left text-[11px]">
                                                <thead className="bg-gray-50 border-b">
                                                    <tr className="text-gray-500 uppercase tracking-wide">
                                                        <th className="py-2 px-3">Index</th>
                                                        <th className="py-2 px-3">Bin / Value</th>
                                                        <th className="py-2 px-3">Label</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-gray-100">
                                                    {tags.map(tag => (
                                                        <tr key={tag.key} className="hover:bg-gray-50">
                                                            <td className="py-1.5 px-3 text-gray-700">
                                                                {tag.key}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-800">
                                                                {tag.bin || tag.value || "—"}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-500">
                                                                {tag.label || tag.description || "—"}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        ) : (
                                            <div className="p-3 text-[11px] text-gray-400">
                                                No composite indices have been stored yet for this image.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Features + validations */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="font-semibold text-[11px] text-gray-700">
                                            Science attributes
                                        </div>
                                        <div className="text-[11px] text-gray-400">
                                            {features.length ? `${features.length} attributes` : "—"}
                                        </div>
                                    </div>
                                    <div className="border rounded-lg max-h-64 overflow-y-auto">
                                        {features.length ? (
                                            <table className="w-full text-left text-[11px]">
                                                <thead className="bg-gray-50 border-b">
                                                    <tr className="text-gray-500 uppercase tracking-wide">
                                                        <th className="py-2 px-3">Key</th>
                                                        <th className="py-2 px-3">Value</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-gray-100">
                                                    {features.map(feat => (
                                                        <tr key={feat.key} className="hover:bg-gray-50">
                                                            <td className="py-1.5 px-3 text-gray-700">
                                                                {feat.key}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-800">
                                                                {feat.value === null || feat.value === undefined
                                                                    ? "—"
                                                                    : feat.value.toFixed
                                                                    ? feat.value.toFixed(3)
                                                                    : feat.value}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        ) : (
                                            <div className="p-3 text-[11px] text-gray-400">
                                                No science attributes recorded yet for this image.
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div>
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="font-semibold text-[11px] text-gray-700">
                                            Human validations
                                        </div>
                                        <div className="text-[11px] text-gray-400">
                                            {inspectorValidations.length
                                                ? `${inspectorValidations.length} validations`
                                                : "—"}
                                        </div>
                                    </div>
                                    <div className="border rounded-lg max-h-64 overflow-y-auto">
                                        {inspectorValidations.length ? (
                                            <table className="w-full text-left text-[11px]">
                                                <thead className="bg-gray-50 border-b">
                                                    <tr className="text-gray-500 uppercase tracking-wide">
                                                        <th className="py-2 px-3">User</th>
                                                        <th className="py-2 px-3">Attribute</th>
                                                        <th className="py-2 px-3">Value</th>
                                                        <th className="py-2 px-3">Dwell (ms)</th>
                                                        <th className="py-2 px-3">When</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-gray-100">
                                                    {inspectorValidations.map(r => (
                                                        <tr key={r.id} className="hover:bg-gray-50">
                                                            <td className="py-1.5 px-3 text-gray-800">
                                                                {r.username || `user-${r.user_id}`}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-600">
                                                                {r.attribute_key}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-600">
                                                                {r.value && r.value.toFixed
                                                                    ? r.value.toFixed(2)
                                                                    : r.value}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-600">
                                                                {r.duration_ms ?? "—"}
                                                            </td>
                                                            <td className="py-1.5 px-3 text-gray-400">
                                                                {r.created_at
                                                                    ? new Date(r.created_at).toLocaleString()
                                                                    : "—"}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        ) : (
                                            <div className="p-3 text-[11px] text-gray-400">
                                                No human validations recorded yet for this image.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}