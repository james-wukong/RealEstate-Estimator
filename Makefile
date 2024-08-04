TEST = pytest
# GOBUILD = $(GOCMD) build

.PHONY: test

test:
	$(TEST) -vv
