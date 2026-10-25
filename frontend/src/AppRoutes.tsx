/**
 * The application's routes and navigation. The Advisor is the home page; pages that
 * moved or were merged redirect, so old links and bookmarks keep working.
 */
import { Navigate, Route, Routes } from "react-router-dom";
import Compare from "./pages/Compare";
import EvalPage from "./pages/Eval";
import Ingest from "./pages/Ingest";
import Advisor from "./pages/Advisor";
import { BeakerIcon, CompareIcon, SparkleIcon, UploadIcon } from "./components/Icons";
import type { MessageKey } from "./i18n";
import { isStaticDemo } from "./staticDemo";

/**
 * Sidebar and mobile navigation, in display order. "main" is the client-facing
 * flow (the Advisor); "lab" holds the technical tools behind it.
 */
export const NAV: {
  to: string;
  label: MessageKey;
  Icon: typeof SparkleIcon;
  end?: boolean;
  group: "main" | "lab";
}[] = [
  { to: "/", label: "nav.advisor", Icon: SparkleIcon, end: true, group: "main" },
  { to: "/ingest", label: "nav.ingest", Icon: UploadIcon, group: "lab" },
  { to: "/compare", label: "nav.compare", Icon: CompareIcon, group: "lab" },
  { to: "/eval", label: "nav.eval", Icon: BeakerIcon, group: "lab" },
];

/** The navigation to show: the static demo has no backend, so no Lab. */
export function visibleNav(): typeof NAV {
  return isStaticDemo() ? NAV.filter((n) => n.group === "main") : NAV;
}

export default function AppRoutes() {
  if (isStaticDemo()) {
    return (
      <Routes>
        <Route path="/" element={<Advisor />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    );
  }
  return (
    <Routes>
      <Route path="/" element={<Advisor />} />
      <Route path="/ask" element={<Navigate to="/compare" replace />} />
      <Route path="/ingest" element={<Ingest />} />
      <Route path="/compare" element={<Compare />} />
      <Route path="/eval" element={<EvalPage />} />
      <Route path="/advisor" element={<Navigate to="/" replace />} />
      <Route path="/dashboard" element={<Navigate to="/eval?tab=overview" replace />} />
      <Route path="/regressions" element={<Navigate to="/eval?tab=runs" replace />} />
      <Route path="/data" element={<Navigate to="/eval" replace />} />
    </Routes>
  );
}
