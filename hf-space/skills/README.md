# Reusable Analytics Agent Skills

These 12 skills are distilled from the supplied codebase while intentionally excluding domain-specific references and internal terminology.

## Skills

1. grounded-data-agent
2. natural-language-ui-controller
3. schema-aware-analytics
4. investigative-analytics-workspace
5. large-scale-analytics-engine
6. llm-provider-orchestration
7. streaming-llm-interface
8. llm-evaluation-feedback
9. automated-report-generation
10. ml-observability
11. human-vs-ai-evaluation
12. enterprise-analytics-ux

Each skill is self-contained in `<skill-name>/SKILL.md`.

The skills are designed to compose:
- grounded-data-agent provides evidence discipline
- natural-language-ui-controller provides agentic UI actions
- schema-aware-analytics provides data-model adaptation
- investigative-analytics-workspace provides the end-user investigation loop
- large-scale-analytics-engine provides scalable retrieval
- llm-provider-orchestration and streaming-llm-interface provide inference infrastructure
- evaluation, reporting, observability, and UX skills provide production operating capabilities
