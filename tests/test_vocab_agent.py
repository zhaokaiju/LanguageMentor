import sys
import os
import unittest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

# 将 src 目录添加到模块搜索路径，方便导入项目中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.vocab_agent import VocabAgent


class TestVocabAgent(unittest.TestCase):

    def setUp(self):
        # 创建临时提示文件
        self.prompt_content = "你是一个词汇学习助手"
        self.prompt_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.prompt_file.write(self.prompt_content)
        self.prompt_file.close()

        # 模拟 langchain 组件
        self.mock_chatbot = MagicMock()
        self.mock_chatbot_with_history = MagicMock()
        self.mock_invoke = MagicMock()
        self.mock_invoke.content = "模拟回复内容"
        self.mock_chatbot_with_history.invoke.return_value = self.mock_invoke

        # 模拟会话历史
        self.mock_history = MagicMock()
        self.mock_clear = MagicMock()
        self.mock_history.clear = self.mock_clear
        self.patcher_history = patch('vocab_agent.get_session_history', return_value=self.mock_history)
        self.patcher_history.start()

        # 创建代理实例
        self.agent = VocabAgent(session_id="test_session")
        self.agent.prompt_file = self.prompt_file.name  # 使用临时文件路径
        self.agent.chatbot_with_history = self.mock_chatbot_with_history  # 注入模拟对象

    def tearDown(self):
        # 清理临时文件
        os.unlink(self.prompt_file.name)
        # 停止所有 patch
        self.patcher_history.stop()

    def test_initialization(self):
        """测试代理初始化是否正确"""
        self.assertEqual(self.agent.name, "vocab_study")
        self.assertEqual(self.agent.session_id, "test_session")
        self.assertEqual(self.agent.prompt, self.prompt_content)
        self.assertIsNone(self.agent.intro_file)
        self.assertEqual(self.agent.intro_messages, [])

    def test_chat_with_history(self):
        """测试带历史记录的聊天功能"""
        # 测试调用
        response = self.agent.chat_with_history("测试输入")

        # 验证调用参数
        self.mock_chatbot_with_history.invoke.assert_called_once()
        args, kwargs = self.mock_chatbot_with_history.invoke.call_args

        # 验证消息内容
        self.assertIsInstance(args[0][0], HumanMessage)
        self.assertEqual(args[0][0].content, "测试输入")

        # 验证会话ID配置
        self.assertEqual(kwargs['configurable']['session_id'], "test_session")

        # 验证返回内容
        self.assertEqual(response, "模拟回复内容")

    def test_restart_session(self):
        """测试重置会话功能"""
        # 调用方法
        result = self.agent.restart_session()

        # 验证历史记录被清除
        self.mock_clear.assert_called_once()

        # 验证返回的是历史对象
        self.assertEqual(result, self.mock_history)

        # 测试自定义 session_id
        custom_session = "custom_session"
        self.agent.restart_session(custom_session)
        self.mock_clear.assert_called()  # 确保再次调用
        self.assertEqual(self.mock_history.clear.call_count, 2)

    def test_prompt_file_not_found(self):
        """测试提示文件不存在时的异常处理"""
        # 删除临时文件以模拟文件不存在
        os.unlink(self.prompt_file.name)

        with self.assertRaises(FileNotFoundError):
            # 重新初始化代理以触发文件读取
            agent = VocabAgent()
            agent.prompt_file = self.prompt_file.name

    @patch('builtins.open', side_effect=Exception("模拟异常"))
    def test_prompt_loading_exception(self, mock_open):
        """测试提示文件加载异常"""
        with self.assertRaises(Exception) as context:
            agent = VocabAgent()
            agent.prompt_file = "invalid_path.txt"

        self.assertEqual(str(context.exception), "模拟异常")

    def test_create_chatbot(self):
        """测试聊天机器人创建过程（需要部分真实对象）"""
        # 使用真实组件测试初始化流程
        agent = VocabAgent(session_id="real_session_test")
        agent.prompt_file = self.prompt_file.name

        # 手动创建聊天机器人
        agent.create_chatbot()

        # 验证关键组件存在
        self.assertIsNotNone(agent.chatbot)
        self.assertIsNotNone(agent.chatbot_with_history)

        # 验证提示模板内容
        self.assertIn(self.prompt_content, str(agent.chatbot))


if __name__ == '__main__':
    unittest.main()
