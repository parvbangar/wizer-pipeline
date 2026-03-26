# enrichment/steps/__init__.py
# Each file in this folder is one enrichment step.
# Steps are pure functions: text/data in → structured result out.
# They have no DB calls and no side effects — the runner handles persistence.
