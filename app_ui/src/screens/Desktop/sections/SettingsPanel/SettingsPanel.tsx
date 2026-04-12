import { useState } from "react";
import type { AppSettings } from "../../../../api";

interface SettingsPanelProps {
  settings: AppSettings;
  models: string[];
  onSave: (patch: Partial<AppSettings>) => Promise<void>;
  onClose: () => void;
}

interface SettingField {
  key: keyof AppSettings;
  label: string;
  description: string;
  type: "range" | "select" | "number";
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  format?: (v: number) => string;
}

const FIELDS: SettingField[] = [
  {
    key: "model_temperature",
    label: "Temperature",
    description: "Controls randomness. Higher = more creative, lower = more deterministic.",
    type: "range",
    min: 0,
    max: 2,
    step: 0.05,
    format: (v) => v.toFixed(2),
  },
  {
    key: "runtime_profile",
    label: "Runtime Profile",
    description: "low_latency skips some checks for speed; balanced is thorough.",
    type: "select",
    options: ["low_latency", "balanced", "high_quality"],
  },
  {
    key: "planner_max_subqueries",
    label: "Max Sub-queries",
    description: "How many retrieval sub-queries the planner generates per user question.",
    type: "range",
    min: 1,
    max: 8,
    step: 1,
    format: (v) => String(Math.round(v)),
  },
  {
    key: "context_chunk_limit",
    label: "Context Chunk Limit",
    description: "Max chunks passed to synthesis. More = richer context but slower.",
    type: "range",
    min: 1,
    max: 10,
    step: 1,
    format: (v) => String(Math.round(v)),
  },
  {
    key: "context_chunk_char_limit",
    label: "Chunk Char Limit",
    description: "Max characters per chunk passed to the LLM.",
    type: "range",
    min: 100,
    max: 1000,
    step: 50,
    format: (v) => String(Math.round(v)),
  },
  {
    key: "max_retrieval_retries",
    label: "Max Retrieval Retries",
    description: "How many times to reformulate and retry if retrieval is weak.",
    type: "range",
    min: 0,
    max: 3,
    step: 1,
    format: (v) => String(Math.round(v)),
  },
  {
    key: "max_validation_retries",
    label: "Max Validation Retries",
    description: "How many times to retry synthesis if citation validation fails.",
    type: "range",
    min: 0,
    max: 3,
    step: 1,
    format: (v) => String(Math.round(v)),
  },
  {
    key: "short_circuit_confidence_threshold",
    label: "Short-Circuit Confidence Threshold",
    description: "If short-circuit confidence drops below this, the system can fall back to full routing.",
    type: "range",
    min: 0,
    max: 1,
    step: 0.05,
    format: (v) => v.toFixed(2),
  },
];

export const SettingsPanel = ({ settings, models, onSave, onClose }: SettingsPanelProps): JSX.Element => {
  const [local, setLocal] = useState<Partial<AppSettings>>({ ...settings });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await onSave(local);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const renderField = (field: SettingField) => {
    const value = local[field.key] ?? settings[field.key];

    if (field.type === "select") {
      return (
        <select
          className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica] bg-slate-50 outline-none focus:border-indigo-400 transition-colors"
          value={String(value)}
          onChange={(e) => setLocal((prev) => ({ ...prev, [field.key]: e.target.value }))}
        >
          {field.options!.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      );
    }

    if (field.type === "range") {
      const numVal = Number(value);
      return (
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-500 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              {field.min}
            </span>
            <span className="text-sm font-bold text-indigo-600 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              {field.format ? field.format(numVal) : numVal}
            </span>
            <span className="text-xs text-slate-500 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              {field.max}
            </span>
          </div>
          <input
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={numVal}
            onChange={(e) =>
              setLocal((prev) => ({ ...prev, [field.key]: Number(e.target.value) as AppSettings[typeof field.key] }))
            }
            className="w-full accent-indigo-600"
          />
        </div>
      );
    }

    return null;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" />
      <div
        className="relative bg-white rounded-3xl shadow-2xl w-[520px] max-h-[85vh] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100">
          <div>
            <h2 className="text-lg font-bold text-slate-800 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Settings
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Changes apply to the current session
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-100 transition-colors text-slate-400 hover:text-slate-600"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-5">
          {/* Model picker */}
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Chat Model
            </label>
            <p className="text-xs text-slate-400 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Ollama model used for planning, synthesis, and validation.
            </p>
            <select
              className="w-full px-3 py-2 rounded-xl border border-slate-200 text-sm text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica] bg-slate-50 outline-none focus:border-indigo-400 transition-colors"
              value={String(local.ollama_chat_model ?? settings.ollama_chat_model)}
              onChange={(e) => setLocal((prev) => ({ ...prev, ollama_chat_model: e.target.value }))}
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div className="border-t border-slate-100" />

          {FIELDS.map((field) => (
            <div key={field.key} className="flex flex-col gap-2">
              <label className="text-sm font-semibold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
                {field.label}
              </label>
              <p className="text-xs text-slate-400 [font-family:'Plus_Jakarta_Sans',Helvetica]">{field.description}</p>
              {renderField(field)}
            </div>
          ))}

          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-slate-700 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Enable Short-Circuit Routing
            </label>
            <p className="text-xs text-slate-400 [font-family:'Plus_Jakarta_Sans',Helvetica]">
              Route simple lookup queries directly for lower latency.
            </p>
            <button
              onClick={() =>
                setLocal((prev) => ({
                  ...prev,
                  enable_short_circuit_routing: !Boolean(
                    prev.enable_short_circuit_routing ?? settings.enable_short_circuit_routing,
                  ),
                }))
              }
              className={`w-full px-3 py-2 rounded-xl border text-sm text-left transition-colors [font-family:'Plus_Jakarta_Sans',Helvetica] ${
                (local.enable_short_circuit_routing ?? settings.enable_short_circuit_routing)
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-slate-50 border-slate-200 text-slate-600"
              }`}
            >
              {(local.enable_short_circuit_routing ?? settings.enable_short_circuit_routing)
                ? "Enabled"
                : "Disabled"}
            </button>
          </div>

          {/* Read-only info */}
          <div className="border-t border-slate-100 pt-4">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider [font-family:'Plus_Jakarta_Sans',Helvetica] mb-3">
              Read-only info
            </p>
            <div className="flex flex-col gap-2">
              {[
                { label: "Chunk size", value: String(settings.chunk_size) },
                { label: "Chunk overlap", value: String(settings.chunk_overlap) },
                { label: "Retrieval top-k", value: String(settings.retrieval_top_k) },
              ].map(({ label, value }) => (
                <div key={label} className="flex justify-between text-xs [font-family:'Plus_Jakarta_Sans',Helvetica]">
                  <span className="text-slate-500">{label}</span>
                  <span className="font-semibold text-slate-700">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between gap-3">
          {saveError && (
            <span className="text-xs text-red-500 [font-family:'Plus_Jakarta_Sans',Helvetica]">{saveError}</span>
          )}
          {!saveError && <div />}
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-sm font-semibold text-slate-600 hover:bg-slate-100 transition-colors [font-family:'Plus_Jakarta_Sans',Helvetica]"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-5 py-2 rounded-xl text-sm font-semibold bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors [font-family:'Plus_Jakarta_Sans',Helvetica] flex items-center gap-2"
            >
              {saving ? (
                <>
                  <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Saving…
                </>
              ) : saved ? (
                "✓ Saved"
              ) : (
                "Save changes"
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
