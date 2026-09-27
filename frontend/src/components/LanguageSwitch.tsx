import { LANGUAGES, useI18n } from "../i18n";

/** EN / FR toggle. The choice is saved in this browser. */
export default function LanguageSwitch({ className = "" }: { className?: string }) {
  const { lang, setLanguage, t } = useI18n();
  return (
    <div
      role="group"
      aria-label={t("lang.label")}
      className={`inline-flex rounded border border-bg-border overflow-hidden text-[11px] font-mono ${className}`}
    >
      {LANGUAGES.map((l) => {
        const active = l === lang;
        return (
          <button
            key={l}
            type="button"
            lang={l}
            onClick={() => setLanguage(l)}
            aria-pressed={active}
            title={t("lang.switchTo", { language: t(`lang.${l}`) })}
            className={`px-2 py-1 uppercase transition-colors ${
              active ? "bg-accent/20 text-white" : "text-zinc-500 hover:text-zinc-200"
            }`}
          >
            {l}
          </button>
        );
      })}
    </div>
  );
}
