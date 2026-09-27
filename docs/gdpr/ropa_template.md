# Registre des activités de traitement / Record of processing activities

> **Modèle, pas un avis juridique. / Template, not legal advice.**
> Ce modèle suit la structure de l'article 30 du RGPD et du modèle de registre proposé par la CNIL. Il est pré-rempli avec ce que fait EvalRAG. Tout ce qui est entre `<…>` dépend de votre déploiement et doit être complété ou validé par le responsable de traitement et son DPO.
> This template follows GDPR art. 30 and the CNIL's record template. It is pre-filled with what EvalRAG does. Everything in `<…>` depends on your deployment and must be completed or validated by the controller and its DPO.

Voir aussi / See also: [GDPR guide](./README.md).

---

## 1. Responsable du traitement / Controller

| Champ / Field | Valeur / Value |
|---|---|
| Organisme / Organisation | `<raison sociale / legal name>` |
| Adresse / Address | `<adresse / address>` |
| SIREN | `<SIREN>` |
| Représentant légal / Legal representative | `<nom, fonction / name, role>` |
| Délégué à la protection des données (DPO) / Data protection officer | `<nom, contact / name, contact>` |
| Représentant dans l'UE (art. 27) / EU representative | `<si applicable / if applicable>` |
| Responsable conjoint / Joint controller | `<si applicable / if applicable>` |

## 2. Activité de traitement / Processing activity

| Champ / Field | Valeur / Value |
|---|---|
| Nom du traitement / Name | Assistant de recherche documentaire (RAG) « EvalRAG » / Document question-answering assistant (RAG) "EvalRAG" |
| Référence interne / Internal reference | `<ex. / e.g. TRT-2026-014>` |
| Date de création / Created | `<date>` |
| Dernière mise à jour / Last updated | `<date>` |
| Service opérationnel / Business owner | `<service / department>` |

## 3. Finalités / Purposes

| # | Finalité / Purpose | Base légale (art. 6) / Lawful basis |
|---|---|---|
| F1 | Indexer des documents et répondre à des questions à partir de leur contenu / Index documents and answer questions from their content | `<ex. intérêt légitime, exécution du contrat / e.g. legitimate interests, contract>` |
| F2 | Évaluer la qualité des réponses (jeux de test, notation) / Evaluate answer quality (golden sets, scoring) | `<ex. intérêt légitime / e.g. legitimate interests>` |
| F3 | Sécurité, contrôle d'accès, comptabilisation de l'usage / Security, access control, usage accounting | `<ex. intérêt légitime, obligation légale / e.g. legitimate interests, legal obligation>` |
| F4 | Diagnostic technique (journaux, traces) / Technical troubleshooting (logs, traces) | `<ex. intérêt légitime / e.g. legitimate interests>` |

Si l'intérêt légitime est retenu, joindre la mise en balance des intérêts. / If relying on legitimate interests, attach the balancing test.

## 4. Catégories de personnes concernées / Categories of data subjects

- [ ] Salariés, agents / Employees, staff
- [ ] Clients, prospects / Customers, prospects
- [ ] Fournisseurs, partenaires / Suppliers, partners
- [ ] Personnes citées dans les documents indexés / People named in indexed documents
- [ ] Utilisateurs de l'assistant (titulaires de clés API) / Assistant users (API key holders)
- [ ] `<autre / other>`

## 5. Catégories de données / Categories of personal data

| Catégorie / Category | Détail / Detail | Concerné ? / Applies? |
|---|---|---|
| Identification | Nom, prénom, e-mail, téléphone (masqués si `PII_MODE_INGEST=mask`) / Name, email, phone (masked if `PII_MODE_INGEST=mask`) | `<oui/non / yes/no>` |
| Numéro de sécurité sociale (NIR) / Social security number | Masqué ou refusé à l'ingestion ; usage encadré en France / Masked or rejected at ingestion; its use is restricted in France | `<oui/non / yes/no>` |
| Données bancaires / Financial | IBAN, cartes (masqués) / IBAN, cards (masked) | `<oui/non / yes/no>` |
| Vie professionnelle / Professional life | Contenu des documents / Document content | `<oui/non / yes/no>` |
| Connexion / Connection data | Adresse IP (journaux), horodatages, identifiant de clé API / IP address (logs), timestamps, API key id | Oui / Yes |
| Contenu libre / Free text | Questions posées, réponses générées, commentaires de notation / Questions asked, generated answers, rating comments | Oui / Yes |
| Données sensibles (art. 9) / Special categories | Santé, opinions, syndicat… / Health, opinions, union membership… | `<à exclure ou justifier / exclude or justify>` |

Les noms de personnes ne sont **pas** détectés automatiquement. / Person names are **not** detected automatically.

## 6. Destinataires / Recipients

| Destinataire / Recipient | Rôle / Role | Données / Data | Activé ? / Enabled? |
|---|---|---|---|
| Équipes internes habilitées / Authorised internal teams | Utilisateurs, administrateurs / Users, admins | Selon habilitations / Per access rights | Oui / Yes |
| Anthropic (API Claude) | Sous-traitant / Processor | Questions, extraits de documents, réponses / Questions, document excerpts, answers | `<oui/non / yes/no>` |
| OpenAI | Sous-traitant / Processor | Questions, extraits / Questions, excerpts | `<oui/non / yes/no>` |
| Langfuse | Sous-traitant / Processor | Traces (masquées) / Traces (masked) | `<oui/non / yes/no>` |
| Weights & Biases | Sous-traitant / Processor | Résultats d'évaluation / Eval results | `<oui/non / yes/no>` |
| Hébergeur / Hosting provider | Sous-traitant / Processor | Toutes les données stockées / All stored data | `<nom / name>` |

Pour chaque sous-traitant : contrat conforme à l'article 28 (DPA) signé ou accepté, date, lien. / For each processor: art. 28 contract (DPA) signed or accepted, date, link.

## 7. Transferts hors UE / Transfers outside the EU

| Destinataire / Recipient | Pays / Country | Garantie (art. 44 s.) / Safeguard | Documentation |
|---|---|---|---|
| Anthropic | `<ex. / e.g. États-Unis / USA>` | `<ex. DPF, clauses contractuelles types / e.g. DPF, SCCs — à vérifier dans le DPA en vigueur / check the current DPA>` | `<lien / link>` |
| `<autre / other>` | | | |

## 8. Durées de conservation / Retention periods

| Donnée / Data | Durée / Period | Paramètre / Setting | Justification |
|---|---|---|---|
| Documents indexés (Qdrant, graphe) / Indexed documents (Qdrant, graph) | `<durée de vie du document source / lifetime of the source document>` | Effacement via `/privacy` / Erasure via `/privacy` | `<…>` |
| Traces de requêtes / Request traces | 7 jours / days (défaut / default) | `RETENTION_TRACES_DAYS` | Diagnostic |
| Résultats d'évaluation / Eval runs | 365 jours / days (défaut / default) | `RETENTION_EVAL_RUNS_DAYS` | Suivi qualité / Quality tracking |
| Journal d'audit / Audit log | 365 jours / days (défaut / default) | `RETENTION_AUDIT_DAYS` | Sécurité / Security |
| Usage par clé API / Per-key usage | `<à définir / to define>` | — | Facturation, quotas / Billing, quotas |
| Journaux applicatifs / Application logs | Rotation 5 × 10 Mo, ou votre politique / 5 × 10 MB rotation, or your policy | `LOG_DIR` | Sécurité, diagnostic / Security, troubleshooting |
| Sauvegardes / Backups | `<…>` | — | `<…>` |

## 9. Mesures de sécurité / Security measures

- [x] Masquage des données personnelles détectées à l'ingestion, dans les journaux et les traces / Masking of detected personal data at ingestion, in logs and in traces
- [x] Clés API stockées sous forme de hachage SHA-256 / API keys stored as SHA-256 hashes
- [x] Limitation de débit par clé / Per-key rate limiting
- [x] Endpoints d'administration protégés par `ADMIN_KEY` / Admin endpoints protected by `ADMIN_KEY`
- [x] Purge automatique selon les durées de conservation / Automatic purge per retention periods
- [ ] `REQUIRE_API_KEY=true` en production / in production
- [ ] Chiffrement des volumes au repos / Encryption of volumes at rest
- [ ] TLS de bout en bout / End-to-end TLS
- [ ] Hébergement UE / EU hosting
- [ ] Sauvegardes chiffrées et testées / Encrypted, tested backups
- [ ] Revue périodique des habilitations / Periodic access review
- [ ] `<autre / other>`

## 10. Droits des personnes / Data subject rights

| Droit / Right | Procédure / Procedure |
|---|---|
| Accès (art. 15) / Access | `GET /privacy/export/{principal_id}`, complété des autres systèmes / supplemented by other systems |
| Effacement (art. 17) / Erasure | `DELETE /privacy/documents/{doc_id}`, `/privacy/sources?source_key=…`, `/privacy/principals/{id}`, plus étapes manuelles du [guide](./README.md#right-to-erasure-art-17) / plus the manual steps in the [guide](./README.md#right-to-erasure-art-17) |
| Rectification (art. 16) | Corriger le document source et le réindexer / Correct the source document and re-ingest it |
| Opposition, limitation (art. 18, 21) / Objection, restriction | `<procédure interne / internal procedure>` |
| Point de contact / Contact point | `<adresse e-mail du DPO / DPO email>` |
| Délai de réponse / Response time | 1 mois (art. 12(3)) / 1 month |

## 11. Analyse d'impact (AIPD) / Impact assessment (DPIA)

| Champ / Field | Valeur / Value |
|---|---|
| AIPD requise ? / DPIA required? | `<oui/non + justification / yes/no + reasoning>` (voir la liste CNIL / see the CNIL list) |
| Référence de l'AIPD / DPIA reference | `<…>` |
| Consultation du CSE (si données salariés) / Works council consulted (employee data) | `<date>` |

## 12. Historique / Change log

| Date | Modification / Change | Auteur / Author |
|---|---|---|
| `<date>` | Création / Created | `<…>` |
