/**
 * The application's routes and navigation. Pages that were merged into the Evaluation page
 * redirect there, so old links and bookmarks keep working.
 */
import { Navigate, Route, Routes } from "react-router-dom";
import Ask from "./pages/Ask";
import Compare from "./pages/Compare";
import EvalPage from "./pages/Eval";
import Ingest from "./pages/Ingest";
import Advisor from "./pages/Advisor";
import { BeakerIcon, ChatIcon, CompareIcon, SparkleIcon, UploadIcon } from "./components/Icons";
import type { MessageKey } from "./i18n";

/** Sidebar and mobile navigation, in display order. */
export const NAV: { to: string; label: MessageKey; Icon: typeof ChatIcon; end?: boolean }[] = [
  { to: "/", label: "nav.ask", Icon: ChatIcon, end: true },
  { to: "/ingest", label: "nav.ingest", Icon: UploadIcon },
  { to: "/compare", label: "nav.compare", Icon: CompareIcon },
  { to: "/eval", label: "nav.eval", Icon: BeakerIcon },
  { to: "/advisor", label: "nav.advisor", Icon: SparkleIcon },
];

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Ask />} />
      <Route path="/ingest" element={<Ingest />} />
      <Route path="/compare" element={<Compare />} />
      <Route path="/eval" element={<EvalPage />} />
      <Route path="/advisor" element={<Advisor />} />
      <Route path="/dashboard" element={<Navigate to="/eval?tab=overview" replace />} />
      <Route path="/regressions" element={<Navigate to="/eval?tab=runs" replace />} />
      <Route path="/data" element={<Navigate to="/eval" replace />} />
    </Routes>
  );
}
