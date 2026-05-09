import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Ask from "./pages/Ask";
import Compare from "./pages/Compare";
import EvalPage from "./pages/Eval";
import Regressions from "./pages/Regressions";
import Ingest from "./pages/Ingest";
import {
  BeakerIcon,
  ChatIcon,
  CompareIcon,
  TrendIcon,
  UploadIcon,
} from "./components/Icons";
import "./index.css";

const qc = new QueryClient();

const NAV = [
  { to: "/", label: "Ask", Icon: ChatIcon, end: true },
  { to: "/ingest", label: "Ingest", Icon: UploadIcon },
  { to: "/compare", label: "Compare", Icon: CompareIcon },
  { to: "/eval", label: "Evaluation", Icon: BeakerIcon },
  { to: "/regressions", label: "Regressions", Icon: TrendIcon },
];

function Wordmark() {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="display text-[22px] font-semibold text-white leading-none">
        Eval<span className="text-accent">·</span>RAG
      </span>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col gap-8 border-r border-bg-border bg-bg-surface px-5 py-6">
      <div className="space-y-1.5">
        <Wordmark />
        <div className="text-[11px] text-zinc-500 font-mono">
          retrieval that grades itself
        </div>
      </div>

      <nav className="flex flex-col gap-0.5">
        <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-600 px-3 mb-2">
          Workspace
        </div>
        {NAV.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-3">
        <div className="text-xs text-zinc-500 leading-relaxed border-l-2 border-bg-border pl-3">
          A RAG system is only as good as the eval that catches it drifting.
          Ingest, ask, score, ship.
        </div>
        <div className="flex items-center justify-between pt-3 border-t border-bg-border text-[11px] text-zinc-500">
          <span className="font-mono">v0.1</span>
          <a
            href="https://github.com/maiabazerji"
            target="_blank"
            rel="noreferrer"
            className="hover:text-zinc-300 transition-colors"
          >
            github →
          </a>
        </div>
      </div>
    </aside>
  );
}

function MobileNav() {
  return (
    <header className="md:hidden flex items-center justify-between px-4 py-3 border-b border-bg-border bg-bg-surface sticky top-0 z-10">
      <Wordmark />
      <nav className="flex gap-1 overflow-x-auto">
        {NAV.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-link whitespace-nowrap !py-1.5 ${isActive ? "active" : ""}`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <MobileNav />
        <main className="flex-1 px-4 sm:px-10 py-10 max-w-5xl w-full mx-auto animate-fade-in">
          {children}
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Ask />} />
            <Route path="/ingest" element={<Ingest />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/eval" element={<EvalPage />} />
            <Route path="/regressions" element={<Regressions />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
