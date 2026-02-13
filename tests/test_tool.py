from __future__ import annotations

import asyncio
import pytest
from typing import Dict, Any

from cogent.core import ToolUse, ToolResult, ToolRegistry, ToolParameter, ToolDefinition
from cogent.core import create_tool_execution_step
from cogent.core import Control, Result
from cogent.core import Env
from cogent.starter import ReActState
from cogent.starter.react import ReActPolicy, ReActConfig

from fakes import FakeModel, FakeTools, FakeMemory


class TestToolParameter:
    """测试ToolParameter类"""
    
    def test_tool_parameter_creation(self):
        """测试创建ToolParameter"""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True
        )
        assert param.name == "query"
        assert param.type == "string"
        assert param.description == "Search query"
        assert param.required == True
        assert param.default is None
    
    def test_tool_parameter_with_default(self):
        """测试创建带默认值的ToolParameter"""
        param = ToolParameter(
            name="limit",
            type="number",
            description="Result limit",
            required=False,
            default=10
        )
        assert param.default == 10


class TestToolDefinition:
    """测试ToolDefinition类"""
    
    def test_tool_definition_creation(self):
        """测试创建ToolDefinition"""
        params = {
            "query": ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            ),
            "limit": ToolParameter(
                name="limit",
                type="number",
                description="Result limit",
                required=False,
                default=10
            )
        }
        
        definition = ToolDefinition(
            name="search",
            description="Search for information",
            parameters=params
        )
        
        assert definition.name == "search"
        assert definition.description == "Search for information"
        assert "query" in definition.parameters
        assert "limit" in definition.parameters
    
    def test_tool_definition_validate_parameters(self):
        """测试参数验证"""
        params = {
            "query": ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            )
        }
        
        definition = ToolDefinition(
            name="search",
            description="Search for information",
            parameters=params
        )
        
        # 验证成功
        assert definition.validate_parameters({"query": "test"}) == True
        # 验证失败（缺少必填参数）
        assert definition.validate_parameters({}) == False


class TestToolRegistry:
    """测试ToolRegistry类"""
    
    def test_tool_registry_creation(self):
        """测试创建ToolRegistry"""
        registry = ToolRegistry()
        assert registry._tools == {}
        assert registry._definitions == {}
    
    def test_tool_registry_register(self):
        """测试注册工具"""
        registry = ToolRegistry()
        
        def test_tool(_env, _state, call):
            return f"Result: {call.args.get('value')}"
        
        registry.register("test", test_tool)
        assert "test" in registry._tools
        assert "test" not in registry._definitions
    
    def test_tool_registry_register_with_definition(self):
        """测试注册带定义的工具"""
        registry = ToolRegistry()
        
        def test_tool(_env, _state, call):
            return f"Result: {call.args.get('value')}"
        
        definition = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "value": ToolParameter(
                    name="value",
                    type="string",
                    description="Test value",
                    required=True
                )
            }
        )
        
        registry.register("test", test_tool, definition)
        assert "test" in registry._tools
        assert "test" in registry._definitions
    
    @pytest.mark.asyncio
    async def test_tool_registry_run(self):
        """测试执行工具"""
        registry = ToolRegistry()
        
        def test_tool(_env, _state, call):
            return f"Result: {call.args.get('value')}"
        
        registry.register("test", test_tool)
        
        call = ToolUse(id="1", name="test", args={"value": "test_value"})
        result = await registry.run(None, None, call)
        
        assert result.id == "1"
        assert result.content == "Result: test_value"
        assert not result.failed
    
    @pytest.mark.asyncio
    async def test_tool_registry_run_with_definition(self):
        """测试执行带定义的工具"""
        registry = ToolRegistry()
        
        def test_tool(_env, _state, call):
            return f"Result: {call.args.get('value')}"
        
        definition = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={
                "value": ToolParameter(
                    name="value",
                    type="string",
                    description="Test value",
                    required=True
                )
            }
        )
        
        registry.register("test", test_tool, definition)
        
        # 测试成功执行
        call = ToolUse(id="1", name="test", args={"value": "test_value"})
        result = await registry.run(None, None, call)
        assert not result.failed
        
        # 测试参数验证失败
        call = ToolUse(id="2", name="test", args={})
        result = await registry.run(None, None, call)
        assert result.failed
        assert "Invalid parameters" in result.content
    
    @pytest.mark.asyncio
    async def test_tool_registry_run_nonexistent_tool(self):
        """测试执行不存在的工具"""
        registry = ToolRegistry()
        
        call = ToolUse(id="1", name="nonexistent", args={})
        result = await registry.run(None, None, call)
        
        assert result.failed
        assert "Tool not found" in result.content
    
    @pytest.mark.asyncio
    async def test_tool_registry_run_with_exception(self):
        """测试执行工具时发生异常"""
        registry = ToolRegistry()
        
        def error_tool(_env, _state, call):
            raise ValueError("Tool error")
        
        registry.register("error", error_tool)
        
        call = ToolUse(id="1", name="error", args={})
        result = await registry.run(None, None, call)
        
        assert result.failed
        assert "Tool error" in result.content


class TestCreateToolExecutionStep:
    """测试create_tool_execution_step函数"""
    
    @pytest.mark.asyncio
    async def test_create_tool_execution_step(self):
        """测试创建和使用工具执行步骤"""
        registry = ToolRegistry()
        
        def test_tool(_env, _state, call):
            return f"Result: {call.args.get('value')}"
        
        registry.register("test", test_tool)
        
        step = create_tool_execution_step(registry)
        
        state = ReActState()
        call = ToolUse(id="1", name="test", args={"value": "test_value"})
        
        # 创建一个简单的env
        fake_model = FakeModel([])
        fake_tools = FakeTools({})
        env = Env(model=fake_model, tools=fake_tools)
        
        result = await step(state, call, env)
        
        assert isinstance(result, Result)
        assert result.control.kind == "continue"
        assert isinstance(result.value, ToolResult)
        assert result.value.content == "Result: test_value"
    
    @pytest.mark.asyncio
    async def test_create_tool_execution_step_error(self):
        """测试工具执行步骤的错误处理"""
        registry = ToolRegistry()
        
        # 不注册工具
        step = create_tool_execution_step(registry)
        
        state = ReActState()
        call = ToolUse(id="1", name="nonexistent", args={})
        
        # 创建一个简单的env
        fake_model = FakeModel([])
        fake_tools = FakeTools({})
        env = Env(model=fake_model, tools=fake_tools)
        
        result = await step(state, call, env)
        
        assert isinstance(result, Result)
        assert result.control.kind == "error"
        assert "Tool not found" in str(result.control.reason)


class TestToolUseHistory:
    """测试工具调用历史记录"""

    @pytest.mark.asyncio
    async def test_react_act_with_tool_call_history(self):
        """测试react_act函数的工具调用历史记录"""
        state = ReActState()
        call = ToolUse(id="1", name="test", args={"value": "test_value"})

        # 创建一个简单的env
        fake_model = FakeModel([])
        fake_tools = FakeTools({"test": lambda args: f"Result: {args['value']}"})
        env = Env(model=fake_model, tools=fake_tools)

        policy = ReActPolicy(ReActConfig())
        result = await policy.act(state, call, env)

        assert isinstance(result, Result)
        assert result.control.kind == "continue"
        assert isinstance(result.value, ToolResult)
        assert result.value.id == "1"
        assert "Result: test_value" in result.value.content

    @pytest.mark.asyncio
    async def test_react_act_with_error(self):
        """测试react_act函数的错误处理和历史记录"""
        state = ReActState()
        call = ToolUse(id="1", name="error", args={"value": "test_value"})

        # 创建一个简单的env，工具会抛出异常
        fake_model = FakeModel([])
        fake_tools = FakeTools({"error": lambda args: 1 / 0})
        env = Env(model=fake_model, tools=fake_tools)

        policy = ReActPolicy(ReActConfig())
        result = await policy.act(state, call, env)

        assert isinstance(result, Result)
        assert result.control.kind == "error"
        assert "Tool execution failed" in str(result.control.reason)


if __name__ == "__main__":
    pytest.main([__file__])
