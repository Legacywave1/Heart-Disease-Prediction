import { useState, useRef } from "react";
import PredictionForm from "./components/PredictionForm"; // Assuming this component exists as before
import {
  Heart,
  Activity,
  AlertCircle,
  Download,
  ArrowUp,
  ArrowDown,
} from "lucide-react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const reportRef = useRef();

  const handlePrediction = (data) => {
    setResult(data);
  };

  const downloadReport = () => {
    const input = reportRef.current;
    html2canvas(input, { scale: 2, useCORS: true }).then((canvas) => {
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
      pdf.save(
        `Heart_Risk_Report_${new Date().toISOString().slice(0, 10)}.pdf`,
      );
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-blue-100">
      {/* --- Header --- */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-red-50 p-2 rounded-lg">
              <Heart className="w-8 h-8 text-red-600 fill-current" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-900">
                CardioRisk AI
              </h1>
              <p className="text-xs text-slate-500 font-medium">
                Clinical Decision Support System
              </p>
            </div>
          </div>

          {result && (
            <button
              onClick={downloadReport}
              className="flex items-center gap-2 bg-slate-900 hover:bg-slate-800 text-white text-sm font-semibold py-2.5 px-5 rounded-lg transition-all shadow-sm active:scale-95"
            >
              <Download className="w-4 h-4" />
              Download PDF
            </button>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* --- LEFT COLUMN: Input Form --- */}
          <div className="lg:col-span-5 bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-bold text-slate-800">
                Patient Vitals
              </h2>
            </div>
            <PredictionForm
              onPredict={handlePrediction}
              setLoading={setLoading}
            />
          </div>

          {/* --- RIGHT COLUMN: Results Report (Captured for PDF) --- */}
          <div className="lg:col-span-7">
            <div
              ref={reportRef}
              className="bg-white rounded-2xl shadow-lg border border-slate-100 p-8 min-h-[600px] flex flex-col justify-between"
            >
              {/* Header of Report */}
              <div className="flex justify-between items-end border-b border-slate-100 pb-6 mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">
                    Clinical Risk Assessment
                  </h2>
                  <p className="text-slate-500 text-sm mt-1">
                    Generated: {new Date().toLocaleDateString()} at{" "}
                    {new Date().toLocaleTimeString()}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">
                    Model Version
                  </div>
                  <div className="badge bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-xs font-bold border border-blue-100">
                    {result ? result.model_status : "Waiting..."}
                  </div>
                </div>
              </div>

              {/* EMPTY STATE */}
              {!result && !loading && (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-12 opacity-60">
                  <div className="bg-slate-50 p-6 rounded-full mb-4">
                    <Activity className="w-12 h-12 text-slate-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-700">
                    Ready for Analysis
                  </h3>
                  <p className="text-slate-500 max-w-xs mt-2">
                    Enter patient data in the left panel to generate a real-time
                    risk profile using XGBoost.
                  </p>
                </div>
              )}

              {/* LOADING STATE */}
              {loading && (
                <div className="flex-1 flex flex-col items-center justify-center">
                  <div className="w-16 h-16 border-4 border-blue-100 border-t-blue-600 rounded-full animate-spin"></div>
                  <p className="mt-6 text-slate-600 font-medium animate-pulse">
                    Running Inference Pipeline...
                  </p>
                </div>
              )}

              {/* RESULTS STATE */}
              {result && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                  {/* 1. Score & Recommendation */}
                  <div
                    className={`rounded-xl p-6 mb-8 border-l-4 ${result.risk_level === "High Risk" ? "bg-red-50 border-red-500" : "bg-green-50 border-green-500"}`}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">
                          Overall Prediction
                        </p>
                        <h3
                          className={`text-3xl font-extrabold ${result.risk_level === "High Risk" ? "text-red-700" : "text-green-700"}`}
                        >
                          {result.risk_level}
                        </h3>
                      </div>
                      <div className="text-right">
                        <span
                          className={`text-5xl font-black ${result.risk_level === "High Risk" ? "text-red-900/10" : "text-green-900/10"}`}
                        >
                          {(result.risk_probability * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <p
                      className={`text-lg font-medium ${result.risk_level === "High Risk" ? "text-red-800" : "text-green-800"}`}
                    >
                      {result.recommendation}
                    </p>
                  </div>

                  {/* 2. Feature Importance (SHAP Analysis) */}
                  <div className="mb-8">
                    <h3 className="text-lg font-bold text-slate-800 mb-5 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-slate-400" />
                      Key Risk Contributors (SHAP)
                    </h3>

                    <div className="space-y-4">
                      {result.top_contributors.map((item, idx) => {
                        // Determine styles based on "impact" from python (increase vs decrease)
                        const isRisk = item.impact === "increase";
                        const barColor = isRisk
                          ? "bg-red-500"
                          : "bg-emerald-500";
                        const textColor = isRisk
                          ? "text-red-700"
                          : "text-emerald-700";
                        const Icon = isRisk ? ArrowUp : ArrowDown;

                        return (
                          <div key={idx} className="group">
                            <div className="flex justify-between text-sm mb-1.5 font-medium">
                              <span className="text-slate-700 flex items-center gap-2">
                                <span className="text-slate-300 font-normal w-4">
                                  {idx + 1}.
                                </span>
                                {item.feature}
                              </span>
                              <span
                                className={`flex items-center gap-1 ${textColor}`}
                              >
                                <Icon className="w-3 h-3" />
                                {isRisk ? "Increases Risk" : "Reduces Risk"}
                              </span>
                            </div>

                            {/* Progress Bar Container */}
                            <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${barColor} transition-all duration-1000 ease-out`}
                                style={{
                                  width: `${Math.min(item.importance * 100 * 2, 100)}%`,
                                }} // Scaling x2 for visibility
                              ></div>
                            </div>
                            <div className="text-right mt-1">
                              <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
                                Impact: {item.importance}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Footer */}
                  <div className="border-t border-slate-100 pt-6 text-center">
                    <p className="text-xs text-slate-400 font-medium">
                      Powered by XGBoost • MLflow Registry • React
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
