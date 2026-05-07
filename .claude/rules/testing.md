Testing rules

Every new module in src/ gets a matching test file in tests/
Test file naming: test_{module_name}.py
Use pytest fixtures for shared setup (DB connections, mock data)
Mock all external API calls — never hit Reddit/YouTube/GNews in tests
Validation notebook results (precision/recall) go in notebooks/, not tests/