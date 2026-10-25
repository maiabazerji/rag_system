import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AppRoutes, { visibleNav } from "./AppRoutes";
import ApiKeyGate from "./components/ApiKeyGate";
import LanguageSwitch from "./components/LanguageSwitch";
import { I18nProvider, useT } from "./i18n";
import { ApiError, RequestTimeoutError } from "./api/client";
import { isStaticDemo } from "./staticDemo";
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
        Veritas<span className="text-accent">·</span>RAG
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

      <NavLinks />

      <div className="mt-auto space-y-3">
        <div className="flex items-center justify-between pt-3 border-t border-bg-border text-[11px] text-zinc-500">
          <span className="font-mono">v0.3</span>
          <LanguageSwitch />
        </div>
      </div>
    </aside>
  );
}

/** The two navigation groups: the client flow, then the technical Lab. */
function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const t = useT();
  const group = (g: "main" | "lab") =>
    visibleNav().filter((n) => n.group === g).map(({ to, label, Icon, end }) => (
      <NavLink
        key={to}
        to={to}
        end={end}
        onClick={onNavigate}
        className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
      >
        <Icon className="w-4 h-4" />
        {t(label)}
      </NavLink>
    ));
  return (
    <nav className="flex flex-col gap-0.5">
      <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500 px-3 mb-2">
        {t("nav.workspace")}
      </div>
      {group("main")}
      {!isStaticDemo() && (
        <>
          <div className="px-3 mt-6 mb-2 flex items-baseline justify-between gap-2">
            <span className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">{t("nav.lab")}</span>
            <span className="text-[10px] text-zinc-600">{t("nav.labHint")}</span>
          </div>
          {group("lab")}
        </>
      )}
    </nav>
  );
}

function MobileNav() {
  const t = useT();
  const [open, setOpen] = React.useState(false);
  return (
    <header className="md:hidden sticky top-0 z-20 border-b border-bg-border bg-bg-surface">
      <div className="flex items-center justify-between gap-2 px-4 py-3">
        <Wordmark />
        <div className="flex items-center gap-2">
          <LanguageSwitch className="shrink-0" />
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? t("nav.closeMenu") : t("nav.menu")}
            className="btn-ghost !px-3 !py-2 min-h-[44px] min-w-[44px]"
          >
            <span aria-hidden="true" className="text-lg leading-none">{open ? "✕" : "☰"}</span>
          </button>
        </div>
      </div>
      {open && (
        <div id="mobile-menu" className="px-3 pb-4">
          <NavLinks onNavigate={() => setOpen(false)} />
        </div>
      )}
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
          {!isStaticDemo() && <ApiKeyGate />}
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
