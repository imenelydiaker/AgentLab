from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "src" / "agentlab" / "benchmarks" / "online_mind2web" / "config.json"

class TestOnlineMind2WebBenchmark:
    def test_model_post_init_loads_tasks(self):
        from agentlab.benchmarks.online_mind2web.benchmark import (
            OnlineMind2WebBenchmark,
        )
        from agentlab.benchmarks.online_mind2web.task import OnlineMind2WebTaskConfig

        benchmark =OnlineMind2WebBenchmark()
        benchmark.task_file = CONFIG_PATH
        benchmark.model_post_init(None)

        assert len(benchmark.env_args_list) == 300

        for env_args in benchmark.env_args_list:
            assert isinstance(env_args.task_config, OnlineMind2WebTaskConfig)
            
        # ensuite at least one of the known task IDs is present
        assert "1223b07536a87e0170ff87cbbebd1d3c" in [env_args.task_config.task_id for env_args in benchmark.env_args_list]

    def test_model_post_init_filters_by_level(self):
        from agentlab.benchmarks.online_mind2web.benchmark import (
            OnlineMind2WebBenchmark,
        )

        benchmark = OnlineMind2WebBenchmark(level="easy")
        benchmark.task_file = CONFIG_PATH
        benchmark.model_post_init(None)

        assert all(env_args.task_config.level == "easy" for env_args in benchmark.env_args_list)
        assert len(benchmark.env_args_list) == 81  
        assert "99daaed9a83c266341d28aa40067d376" in [env_args.task_config.task_id for env_args in benchmark.env_args_list]
    
    def test_model_post_init_uses_default_file(self):
        from agentlab.benchmarks.online_mind2web.benchmark import (
            OnlineMind2WebBenchmark,
        )

        benchmark = OnlineMind2WebBenchmark()
        benchmark.model_post_init(None)

        assert len(benchmark.env_args_list) == 300

    def test_model_post_init_handles_missing_file(self, caplog):
        from agentlab.benchmarks.online_mind2web.benchmark import (
            OnlineMind2WebBenchmark,
        )

        benchmark = OnlineMind2WebBenchmark()
        benchmark.task_file = "non_existent_file.json"

        with caplog.at_level("WARNING"):
            benchmark.model_post_init(None)

        assert "Task file not found" in caplog.text