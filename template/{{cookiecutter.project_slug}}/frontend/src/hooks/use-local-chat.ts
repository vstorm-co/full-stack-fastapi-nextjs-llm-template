"use client";

import { useCallback, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { useWebSocket } from "./use-websocket";
import { useLocalChatStore } from "@/stores/local-chat-store";
import type { ChatMessage, ToolCall, WSEvent } from "@/types";
import { WS_URL } from "@/lib/constants";

export function useLocalChat() {
  const {
    currentConversationId,
    getCurrentMessages,
    createConversation,
    addMessage,
    updateMessage,
    addToolCall,
    updateToolCall,
    clearCurrentMessages,
  } = useLocalChatStore();

  const messages = getCurrentMessages();
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentMessageId, setCurrentMessageId] = useState<string | null>(null);
  // Use ref for groupId to avoid React state timing issues with rapid WebSocket events
  const currentGroupIdRef = useRef<string | null>(null);

  const handleWebSocketMessage = useCallback(
    (event: MessageEvent) => {
      const wsEvent: WSEvent = JSON.parse(event.data);

      // Helper to create a new message for CrewAI events
      const createNewMessage = (content: string): string => {
        // Mark previous message as not streaming before creating new one
        if (currentMessageId) {
          updateMessage(currentMessageId, (msg) => ({
            ...msg,
            isStreaming: false,
          }));
        }

        const newMsgId = nanoid();
        addMessage({
          id: newMsgId,
          role: "assistant",
          content,
          timestamp: new Date(),
          isStreaming: true,
          toolCalls: [],
          groupId: currentGroupIdRef.current || undefined,
        });
        setCurrentMessageId(newMsgId);
        return newMsgId;
      };

      switch (wsEvent.type) {
        case "model_request_start": {
          // PydanticAI/LangChain - create message immediately
          createNewMessage("");
          break;
        }

        case "crew_start":
        case "crew_started": {
          // CrewAI - generate groupId for this execution, wait for agent events
          currentGroupIdRef.current = nanoid();
          break;
        }

        case "text_delta": {
          if (currentMessageId) {
            const content = (wsEvent.data as { index: number; content: string })
              .content;
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + content,
            }));
          }
          break;
        }

        // CrewAI agent events - each agent gets its own message container
        case "agent_started": {
          const { agent } = wsEvent.data as {
            agent: string;
            task: string;
          };
          // Create NEW message for this agent (groupId read from ref)
          createNewMessage(`🤖 **${agent}** is starting...`);
          break;
        }

        case "agent_completed": {
          // Finalize current agent's message with output
          if (currentMessageId) {
            const { agent, output } = wsEvent.data as {
              agent: string;
              output: string;
            };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: `✅ **${agent}**\n\n${output}`,
              isStreaming: false,
            }));
          }
          break;
        }

        // CrewAI task events - create separate message for each task
        case "task_started": {
          const { description, agent } = wsEvent.data as {
            task_id: string;
            description: string;
            agent: string;
          };
          // Create NEW message for this task (groupId read from ref)
          createNewMessage(`📋 **Task** (${agent})\n\n${description}`);
          break;
        }

        case "task_completed": {
          // Finalize the task message
          if (currentMessageId) {
            const { output, agent } = wsEvent.data as {
              task_id: string;
              output: string;
              agent: string;
            };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: `✅ **Task completed** (${agent})\n\n${output}`,
              isStreaming: false,
            }));
          }
          break;
        }

        // CrewAI tool events - add as tool calls to current message
        case "tool_started": {
          if (currentMessageId) {
            const { tool_name, tool_args, agent } = wsEvent.data as {
              tool_name: string;
              tool_args: string;
              agent: string;
            };
            const toolCall: ToolCall = {
              id: nanoid(),
              name: tool_name,
              args: { input: tool_args, agent },
              status: "running",
            };
            addToolCall(currentMessageId, toolCall);
          }
          break;
        }

        case "tool_finished": {
          // Tool finished - update last tool call status
          if (currentMessageId) {
            const { tool_name, tool_result } = wsEvent.data as {
              tool_name: string;
              tool_result: string;
              agent: string;
            };
            // Find and update the matching tool call
            updateMessage(currentMessageId, (msg) => {
              const toolCalls = msg.toolCalls || [];
              const lastToolCall = toolCalls.find(
                (tc) => tc.name === tool_name && tc.status === "running"
              );
              if (lastToolCall) {
                return {
                  ...msg,
                  toolCalls: toolCalls.map((tc) =>
                    tc.id === lastToolCall.id
                      ? { ...tc, result: tool_result, status: "completed" as const }
                      : tc
                  ),
                };
              }
              return msg;
            });
          }
          break;
        }

        // LLM events - silently ignored
        case "llm_started":
        case "llm_completed": {
          break;
        }

        case "tool_call": {
          if (currentMessageId) {
            const { tool_name, args, tool_call_id } = wsEvent.data as {
              tool_name: string;
              args: Record<string, unknown>;
              tool_call_id: string;
            };
            const toolCall: ToolCall = {
              id: tool_call_id,
              name: tool_name,
              args,
              status: "running",
            };
            addToolCall(currentMessageId, toolCall);
          }
          break;
        }

        case "tool_result": {
          if (currentMessageId) {
            const { tool_call_id, content } = wsEvent.data as {
              tool_call_id: string;
              content: string;
            };
            updateToolCall(currentMessageId, tool_call_id, {
              result: content,
              status: "completed",
            });
          }
          break;
        }

        case "final_result": {
          if (currentMessageId) {
            const { output } = wsEvent.data as { output: string };
            // For CrewAI, replace content with final output if it exists
            if (output) {
              updateMessage(currentMessageId, (msg) => ({
                ...msg,
                content: msg.content || output,
                isStreaming: false,
              }));
            } else {
              updateMessage(currentMessageId, (msg) => ({
                ...msg,
                isStreaming: false,
              }));
            }
          }
          setIsProcessing(false);
          setCurrentMessageId(null);
          currentGroupIdRef.current = null;
          break;
        }

        case "error": {
          if (currentMessageId) {
            const { message } = wsEvent.data as { message: string };
            updateMessage(currentMessageId, (msg) => ({
              ...msg,
              content: msg.content + `\n\n❌ Error: ${message || "Unknown error"}`,
              isStreaming: false,
            }));
          }
          setIsProcessing(false);
          break;
        }

        case "complete": {
          setIsProcessing(false);
          break;
        }
      }
    },
    [currentMessageId, addMessage, updateMessage, addToolCall, updateToolCall]
  );

  const wsUrl = `${WS_URL}/api/v1/ws/agent`;

  const { isConnected, connect, disconnect, sendMessage } = useWebSocket({
    url: wsUrl,
    onMessage: handleWebSocketMessage,
  });

  const sendChatMessage = useCallback(
    (content: string) => {
      let convId = currentConversationId;
      if (!convId) {
        convId = createConversation();
      }

      const userMessage: ChatMessage = {
        id: nanoid(),
        role: "user",
        content,
        timestamp: new Date(),
      };
      addMessage(userMessage);

      setIsProcessing(true);
      sendMessage({ message: content });
    },
    [addMessage, sendMessage, currentConversationId, createConversation]
  );

  const startNewChat = useCallback(() => {
    createConversation();
  }, [createConversation]);

  return {
    messages,
    currentConversationId,
    isConnected,
    isProcessing,
    connect,
    disconnect,
    sendMessage: sendChatMessage,
    clearMessages: clearCurrentMessages,
    startNewChat,
  };
}
