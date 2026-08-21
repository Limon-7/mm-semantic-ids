# Pipeline entry points. RS_DATA must point at the data folder.
#
#   make data embed CATEGORY=video_games
#   make train EXP=tiger ARGS="name=vg_lastout_LF data=video_games ids.spec.variant=LF"
#   make smoke

CATEGORY ?= video_games
EXP ?= tiger
ARGS ?=
GPU ?= 0
RUN = CUDA_VISIBLE_DEVICES=$(GPU) uv run python -m

.PHONY: data embed tokenise train matrix tables smoke test lint demo

data:
	$(RUN) mmsid.data.download --category $$(awk '/^category:/ {print $$2}' configs/data/$(CATEGORY).yaml)
	$(RUN) mmsid.data.preprocess --config configs/data/$(CATEGORY).yaml
	$(RUN) mmsid.data.images --name $(CATEGORY)
	$(RUN) mmsid.data.audit --name $(CATEGORY)

embed:
	$(RUN) mmsid.embed.text --name $(CATEGORY) --config configs/embed/sent_t5.yaml
	$(RUN) mmsid.embed.image --name $(CATEGORY) --config configs/embed/siglip.yaml

tokenise:
	$(RUN) mmsid.train_tokenizer data=$(CATEGORY) $(ARGS)

train:
	$(RUN) mmsid.train_seq $(EXP) $(ARGS)

matrix:
	scripts/run_matrix.sh $(GPU)

tables:
	uv run python scripts/make_results_tables.py

smoke:
	$(MAKE) data embed CATEGORY=all_beauty
	rm -rf results/smoke
	$(RUN) mmsid.train_seq smoke

test:
	uv run pytest

lint:
	uv run ruff check .

demo:
	uv run python demo/app.py
