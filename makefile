clean:
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist runs rmdir /s /q runs
else
	rm -rf __pycache__ runs
endif

clean_pycache:
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ rmdir /s /q __pycache__
else
	rm -rf __pycache__
endif

clean_runs:
ifeq ($(OS),Windows_NT)
	@if exist runs rmdir /s /q runs
else
	rm -rf runs
endif
