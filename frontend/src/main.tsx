import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AppRoutes, { NAV } from "./AppRoutes";
import ApiKeyGate from "./components/ApiKeyGate";
import LanguageSwitch from "./components/LanguageSwitch";
import { I18nProvider, useT } from "./i18n";
import { ApiError, RequestTimeoutError } from "./api/client";
import "./index.css";

const qc = new QueryClient({
  defaultOptions: {
    queries: {
      // A 4xx (bad key, forbidden, not found) or a timeout will not fix itself
      // on retry; a 5xx or a network blip might.
      retry: (failureCount, error) => {
        if (error instanceof RequestTimeoutError) return false;
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
        return failureCount < 2;
      },
    },
  },
});

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
  const t = useT();
  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col gap-8 border-r border-bg-border bg-bg-surface px-5 py-6">
      <div className="space-y-1.5">
        <Wordmark />
        <div className="text-[11px] text-zinc-500 font-mono">
          {t("nav.tagline")}
        </div>
      </div>

      <nav className="flex flex-col gap-0.5">
        <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-600 px-3 mb-2">
          {t("nav.workspace")}
        </div>
        {NAV.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Icon className="w-4 h-4" />
            {t(label)}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-3">
        <div className="text-xs text-zinc-500 leading-relaxed border-l-2 border-bg-border pl-3">
          {t("nav.blurb")}
        </div>
        <div className="flex items-center justify-between pt-3 border-t border-bg-border text-[11px] text-zinc-500">
          <span className="font-mono">v0.3</span>
          <LanguageSwitch />
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
  const t = useT();
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
            {t(label)}
          </NavLink>
        ))}
      </nav>
      <LanguageSwitch className="ml-2 shrink-0" />
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
          <ApiKeyGate />
          {children}
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <I18nProvider>
      <QueryClientProvider client={qc}>
        <BrowserRouter>
          <Layout>
            <AppRoutes />
          </Layout>
        </BrowserRouter>
      </QueryClientProvider>
    </I18nProvider>
  </React.StrictMode>
);
