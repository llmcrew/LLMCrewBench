PY ?= .venv/bin/python
UV ?= uv

.PHONY: help setup browser agent sandbox agent-image serve test ci docker-smoke cov selfcheck list generate run bench report publish doctor preflight preflight-sandbox visual shots clean

help:
	@echo "setup      создать .venv (uv) и поставить pytest/pytest-cov"
	@echo "browser    поставить playwright + Chromium для визуального блока (категория K)"
	@echo "agent      поставить pi — агентный рантайм категорий C/K/L"
	@echo "            (--no-agent у run/bench оставляет C/K/L на чат-пути)"
	@echo "test       pytest по tests/"
	@echo "ci         то же, что в CI: compileall + pytest (без docker и отчётов)"
	@echo "agent-image  собрать образ с pinned pi для агентных прогонов в контейнере"
	@echo "sandbox    проверить, что контейнер поднимается (образ, docker, pi)"
	@echo "preflight-sandbox  собрать (один раз) и проверить образ песочницы без конфигурации"
	@echo "docker-smoke  проверить контракт песочницы (tests/test_sandbox.py, нужен docker)"
	@echo "cov        pytest с покрытием dailybench/"
	@echo "selfcheck  эталон/мусор без GPU — доказательство, что генераторы и evaluator'ы согласованы"
	@echo "list       показать задания сьюта (--only A,B для подмножества)"
	@echo "generate   записать prompts/oracles/digest сьюта в suite/<seed>"
	@echo "doctor     проверить GPU, toolchain и endpoint"
	@echo "run        прогон на модели: make run ARGS='--config <id> --speed'"
	@echo "bench      один запуск целиком: preflight -> run -> отчёт"
	@echo "report     пересобрать отчёты: make report DIRS='runs/a runs/b' [COMPARE='метка,метка']"
	@echo "publish    собрать отчёт-артефакт в reports/ (коммитится): make publish DIRS='runs/a runs/b'"
	@echo "preflight  проверить, что runtime совпадает с эталонной конфигурацией: make preflight CFG=<id>"
	@echo "visual     отрисовать одну страницу визуального блока: make visual ARGS='/tmp/app.html --steps 90'"
	@echo "serve      локальный сервер для отчёта (живые сцены требуют http): make serve DIR=runs"
	@echo "shots      показать кадры последнего прогона: make shots DIR=runs/<name>"

setup:
	@test -x $(PY) || $(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PY) -e ".[dev]"

browser:
	$(UV) pip install --python $(PY) -e ".[dev,visual]"
	$(PY) -m playwright install chromium
	$(PY) -m dailybench doctor

agent:
	$(PY) -c "from dailybench import agent; print(agent.ensure(lambda m: print(m)))"
	$(PY) -m dailybench doctor

test:
	$(PY) -m pytest tests

ci:
	$(PY) -m compileall -q dailybench scripts tests video/tools
	$(PY) -m pytest tests

agent-image:
	$(PY) scripts/sandbox_run.py build

sandbox:
	$(PY) -c "from dailybench import isolation; ok, reason = isolation.ready(); print(reason or 'контейнер доступен'); raise SystemExit(0 if ok else 2)"

docker-smoke:
	$(PY) -m pytest tests/test_sandbox.py

cov:
	$(PY) -m pytest tests --cov=dailybench --cov-report=term-missing

selfcheck:
	$(PY) -m dailybench selfcheck $(if $(ONLY),--only $(ONLY),)

list:
	$(PY) -m dailybench list $(if $(ONLY),--only $(ONLY),)

generate:
	$(PY) -m dailybench generate --out suite/$$($(PY) -c "from dailybench.config import DEFAULT_SEED; print(DEFAULT_SEED)")

doctor:
	$(PY) -m dailybench doctor $(if $(CFG),--config $(CFG),)

run:
	$(PY) -m dailybench run $(ARGS)

bench:
	$(PY) -m dailybench bench $(ARGS)

report:
	$(PY) -m dailybench report $(DIRS) $(if $(COMPARE),--compare "$(COMPARE)",)

publish:
	$(PY) -m dailybench publish $(DIRS) $(if $(OUT),--out $(OUT),)

preflight:
	$(PY) -m dailybench preflight --config $(CFG) --full

preflight-sandbox:
	$(PY) -m dailybench preflight sandbox

serve:
	@echo "открой http://127.0.0.1:8765/report.html (отчёт прогона)"
	@echo "или   http://127.0.0.1:8765/comparison.html (сравнение, если он есть)"
	cd $(DIR) && $(PY) -m http.server 8765 --bind 127.0.0.1

visual:
	$(PY) -m dailybench visual $(ARGS)

shots:
	@test -f $(DIR)/visual.html || { echo "в $(DIR) нет витрины: прогон без визуального блока"; exit 1; }
	@echo "открой $(DIR)/visual.html"

clean:
	rm -rf .pytest_cache .coverage
