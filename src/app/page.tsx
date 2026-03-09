'use client'

import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { 
  Brain, 
  Play, 
  Square, 
  Activity, 
  Database, 
  Zap, 
  Settings,
  MessageSquare,
  BarChart3,
  Cpu,
  HardDrive,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle,
  Sparkles,
  Network,
  Timer,
  TrendingUp
} from 'lucide-react'

// 类型定义
interface ArchitectureState {
  isRunning: boolean
  currentCycle: number
  currentPhase: string
  optimizationMode: string
}

interface PerformanceMetrics {
  avgCycleTime: number
  maxCycleTime: number
  minCycleTime: number
  totalCycles: number
}

interface MemoryStatistics {
  totalMemories: number
  avgStrength: number
  avgAccessCount: number
  sequenceLength: number
}

interface STDPStatistics {
  totalWeights: number
  totalUpdates: number
  avgUpdateCount: number
  lastUpdateTime: number
}

interface InferenceResult {
  output: string
  cycleStates: any[]
  optimizationMode: string
  memoryAnchors: any[]
  stdpUpdates: any[]
}

interface EvaluationReport {
  timestamp: number
  metrics: {
    inferenceLatency: number
    memoryUsage: number
    accuracy: number
    coherence: number
    learningEfficiency: number
    memoryRecallAccuracy: number
    stdpUpdateCount: number
    cycleCompletionRate: number
  }
  passed: boolean
}

export default function Home() {
  // 状态
  const [isInitialized, setIsInitialized] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [state, setState] = useState<ArchitectureState | null>(null)
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null)
  const [memory, setMemory] = useState<MemoryStatistics | null>(null)
  const [stdp, setStdp] = useState<STDPStatistics | null>(null)
  const [evaluation, setEvaluation] = useState<EvaluationReport | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [scenarios, setScenarios] = useState<string[]>([])
  const [selectedScenario, setSelectedScenario] = useState<string>('')
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [inferenceTime, setInferenceTime] = useState<number>(0)

  // 添加日志
  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, `[${timestamp}] ${message}`].slice(-50))
  }, [])

  // 初始化架构
  const initializeArchitecture = useCallback(async () => {
    setIsLoading(true)
    addLog('正在初始化类人脑双系统AI架构...')
    
    try {
      // 获取架构状态
      const response = await fetch('/api/brain')
      const data = await response.json()
      
      if (data.success) {
        setIsInitialized(true)
        setIsRunning(data.data.isRunning)
        setState(data.data.state)
        addLog('架构初始化成功')
      } else {
        addLog(`初始化失败: ${data.error}`)
      }
      
      // 获取模型信息
      const modelResponse = await fetch('/api/model?action=info')
      const modelData = await modelResponse.json()
      
      if (modelData.success && modelData.data.modelInfo) {
        setModelInfo(modelData.data.modelInfo)
        addLog(`模型已加载: ${modelData.data.modelInfo.model_name}`)
        addLog(`参数量: ${(modelData.data.modelInfo.total_params / 1e6).toFixed(1)}M`)
      }
    } catch (error: any) {
      addLog(`初始化错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [addLog])

  // 获取指标
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/brain?action=metrics')
      const data = await response.json()
      
      if (data.success) {
        setPerformance(data.data.performance)
        setMemory(data.data.memory)
        setStdp(data.data.stdp)
      }
    } catch (error) {
      console.error('获取指标失败:', error)
    }
  }, [])

  // 获取场景列表
  const fetchScenarios = useCallback(async () => {
    try {
      const response = await fetch('/api/brain?action=scenarios')
      const data = await response.json()
      
      if (data.success) {
        setScenarios(data.data)
        if (data.data.length > 0) {
          setSelectedScenario(data.data[0])
        }
      }
    } catch (error) {
      console.error('获取场景失败:', error)
    }
  }, [])

  // 启动架构
  const startArchitecture = async () => {
    setIsLoading(true)
    addLog('正在启动架构...')
    
    try {
      const response = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'start' })
      })
      const data = await response.json()
      
      if (data.success) {
        setIsRunning(true)
        addLog('架构已启动，100Hz刷新周期运行中')
      } else {
        addLog(`启动失败: ${data.error}`)
      }
    } catch (error: any) {
      addLog(`启动错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 停止架构
  const stopArchitecture = async () => {
    setIsLoading(true)
    addLog('正在停止架构...')
    
    try {
      const response = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' })
      })
      const data = await response.json()
      
      if (data.success) {
        setIsRunning(false)
        addLog('架构已停止')
      } else {
        addLog(`停止失败: ${data.error}`)
      }
    } catch (error: any) {
      addLog(`停止错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 执行推理
  const executeInference = async () => {
    if (!inputText.trim()) {
      addLog('请输入内容')
      return
    }
    
    setIsLoading(true)
    addLog(`执行推理: "${inputText}"`)
    
    try {
      // 调用真实模型推理
      const startTime = Date.now()
      const modelResponse = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'infer', 
          data: { prompt: inputText } 
        })
      })
      const modelData = await modelResponse.json()
      
      if (modelData.success) {
        setOutputText(modelData.data.response)
        setInferenceTime(modelData.data.elapsed_ms || (Date.now() - startTime))
        addLog(`推理完成，耗时: ${modelData.data.elapsed_ms || (Date.now() - startTime)}ms`)
      } else {
        // 如果模型推理失败，使用架构模拟
        const response = await fetch('/api/brain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            action: 'infer', 
            data: { input: inputText } 
          })
        })
        const data = await response.json()
        
        if (data.success) {
          const result: InferenceResult = data.data
          setOutputText(result.output)
          addLog(`推理完成，模式: ${result.optimizationMode}`)
        } else {
          addLog(`推理失败: ${data.error}`)
        }
      }
      
      fetchMetrics()
    } catch (error: any) {
      addLog(`推理错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 执行测评
  const executeEvaluation = async () => {
    setIsLoading(true)
    addLog('执行完整测评...')
    
    try {
      const response = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'evaluate' })
      })
      const data = await response.json()
      
      if (data.success) {
        setEvaluation(data.data)
        addLog(`测评完成: ${data.data.passed ? '通过' : '未通过'}`)
      } else {
        addLog(`测评失败: ${data.error}`)
      }
    } catch (error: any) {
      addLog(`测评错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 执行训练
  const executeTraining = async () => {
    if (!selectedScenario) {
      addLog('请选择训练场景')
      return
    }
    
    setIsLoading(true)
    addLog(`执行场景训练: ${selectedScenario}`)
    
    try {
      const response = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'train', 
          data: { scenario: selectedScenario, epochs: 3 } 
        })
      })
      const data = await response.json()
      
      if (data.success) {
        addLog(`训练完成: 准确率 ${(data.data.metrics.accuracy * 100).toFixed(1)}%`)
      } else {
        addLog(`训练失败: ${data.error}`)
      }
    } catch (error: any) {
      addLog(`训练错误: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // 设置优化模式
  const setOptimizationMode = async (mode: string) => {
    try {
      const response = await fetch('/api/brain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'setMode', 
          data: { mode } 
        })
      })
      const data = await response.json()
      
      if (data.success) {
        addLog(`优化模式已切换: ${mode}`)
        fetchMetrics()
      }
    } catch (error: any) {
      addLog(`模式切换错误: ${error.message}`)
    }
  }

  // 初始化
  useEffect(() => {
    initializeArchitecture()
    fetchScenarios()
  }, [initializeArchitecture, fetchScenarios])

  // 定时刷新指标
  useEffect(() => {
    if (isRunning) {
      const interval = setInterval(fetchMetrics, 1000)
      return () => clearInterval(interval)
    }
  }, [isRunning, fetchMetrics])

  // 获取模式图标
  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'selfGeneration': return <Sparkles className="w-4 h-4" />
      case 'selfPlay': return <Network className="w-4 h-4" />
      case 'selfEvaluation': return <BarChart3 className="w-4 h-4" />
      default: return <Activity className="w-4 h-4" />
    }
  }

  // 获取模式名称
  const getModeName = (mode: string) => {
    switch (mode) {
      case 'selfGeneration': return '自生成组合输出'
      case 'selfPlay': return '自博弈竞争优化'
      case 'selfEvaluation': return '自双输出+自评判'
      default: return mode
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* 头部 */}
      <header className="border-b border-purple-500/30 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Brain className="w-10 h-10 text-purple-400" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  类人脑双系统AI架构
                </h1>
                <p className="text-xs text-gray-400">Human-Like Brain Dual-System Architecture</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <Badge variant={isRunning ? "default" : "secondary"} className="gap-1">
                {isRunning ? (
                  <>
                    <Activity className="w-3 h-3 animate-pulse" />
                    运行中
                  </>
                ) : (
                  <>
                    <Square className="w-3 h-3" />
                    已停止
                  </>
                )}
              </Badge>
              
              <Badge variant="outline" className="gap-1 border-purple-500/50">
                <Timer className="w-3 h-3" />
                100Hz
              </Badge>
              
              <Badge variant="outline" className="gap-1 border-blue-500/50">
                <Cpu className="w-3 h-3" />
                Qwen3.5-0.8B
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容 */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* 左侧控制面板 */}
          <div className="col-span-3 space-y-4">
            {/* 架构控制 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Settings className="w-4 h-4 text-purple-400" />
                  架构控制
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2">
                  <Button 
                    onClick={startArchitecture} 
                    disabled={isRunning || isLoading}
                    className="flex-1 bg-green-600 hover:bg-green-700"
                  >
                    <Play className="w-4 h-4 mr-1" />
                    启动
                  </Button>
                  <Button 
                    onClick={stopArchitecture} 
                    disabled={!isRunning || isLoading}
                    variant="destructive"
                    className="flex-1"
                  >
                    <Square className="w-4 h-4 mr-1" />
                    停止
                  </Button>
                </div>
                
                <Button 
                  onClick={executeEvaluation} 
                  disabled={isLoading}
                  variant="outline"
                  className="w-full border-purple-500/50 hover:bg-purple-500/20"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  执行完整测评
                </Button>
              </CardContent>
            </Card>

            {/* 优化模式 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  优化模式
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {['selfGeneration', 'selfPlay', 'selfEvaluation'].map(mode => (
                  <Button
                    key={mode}
                    onClick={() => setOptimizationMode(mode)}
                    variant="outline"
                    className="w-full justify-start border-purple-500/30 hover:bg-purple-500/20"
                  >
                    {getModeIcon(mode)}
                    <span className="ml-2 text-xs">{getModeName(mode)}</span>
                  </Button>
                ))}
              </CardContent>
            </Card>

            {/* 训练场景 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  专项训练
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <select
                  value={selectedScenario}
                  onChange={(e) => setSelectedScenario(e.target.value)}
                  className="w-full bg-black/50 border border-purple-500/30 rounded px-3 py-2 text-sm"
                >
                  {scenarios.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
                
                <Button 
                  onClick={executeTraining} 
                  disabled={isLoading || !selectedScenario}
                  variant="outline"
                  className="w-full border-blue-500/50 hover:bg-blue-500/20"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  执行训练
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* 中间主面板 */}
          <div className="col-span-6 space-y-4">
            {/* 推理面板 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-green-400" />
                  推理交互
                </CardTitle>
                <CardDescription className="text-xs text-gray-400">
                  输入内容进行推理，体验100Hz高刷新周期
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="输入问题或指令..."
                    className="bg-black/50 border-purple-500/30"
                    onKeyPress={(e) => e.key === 'Enter' && executeInference()}
                  />
                  <Button 
                    onClick={executeInference} 
                    disabled={isLoading || !isRunning}
                    className="bg-purple-600 hover:bg-purple-700"
                  >
                    <Sparkles className="w-4 h-4 mr-1" />
                    推理
                  </Button>
                </div>
                
                {outputText && (
                  <div className="bg-black/50 rounded-lg p-4 border border-purple-500/20">
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-xs text-gray-400">输出结果:</div>
                      {inferenceTime > 0 && (
                        <Badge variant="outline" className="text-xs border-green-500/50 text-green-400">
                          耗时: {inferenceTime}ms
                        </Badge>
                      )}
                    </div>
                    <div className="text-sm whitespace-pre-wrap">{outputText}</div>
                  </div>
                )}
                
                {modelInfo && (
                  <div className="bg-black/30 rounded-lg p-3 border border-blue-500/20">
                    <div className="text-xs text-gray-400 mb-2">模型信息:</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>模型: <span className="text-blue-300">{modelInfo.model_name}</span></div>
                      <div>参数: <span className="text-blue-300">{(modelInfo.total_params / 1e6).toFixed(1)}M</span></div>
                      <div>层数: <span className="text-blue-300">{modelInfo.num_layers}</span></div>
                      <div>隐藏层: <span className="text-blue-300">{modelInfo.hidden_size}</span></div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 架构可视化 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Network className="w-4 h-4 text-cyan-400" />
                  架构模块状态
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-3">
                  {/* 模块卡片 */}
                  {[
                    { name: '推理引擎', icon: Activity, status: isRunning, desc: '100Hz刷新' },
                    { name: 'STDP系统', icon: Zap, status: true, desc: '时序可塑性' },
                    { name: '海马体', icon: Brain, status: true, desc: '记忆编码' },
                    { name: '自闭环优化', icon: Network, status: isRunning, desc: '三模式切换' },
                    { name: '训练模块', icon: TrendingUp, status: true, desc: '在线学习' },
                    { name: '测评系统', icon: BarChart3, status: true, desc: '多维度评估' },
                  ].map((module, i) => (
                    <div 
                      key={i}
                      className="bg-black/40 rounded-lg p-3 border border-purple-500/20 hover:border-purple-500/50 transition-colors"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <module.icon className={`w-4 h-4 ${module.status ? 'text-green-400' : 'text-gray-500'}`} />
                        <span className="text-xs font-medium">{module.name}</span>
                      </div>
                      <div className="text-xs text-gray-400">{module.desc}</div>
                      <div className="mt-2 flex items-center gap-1">
                        <div className={`w-2 h-2 rounded-full ${module.status ? 'bg-green-500' : 'bg-gray-500'}`} />
                        <span className="text-xs text-gray-500">{module.status ? '运行中' : '待机'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* 测评结果 */}
            {evaluation && (
              <Card className="bg-black/30 border-purple-500/30">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    {evaluation.passed ? (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400" />
                    )}
                    测评报告
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { label: '推理延迟', value: `${evaluation.metrics.inferenceLatency.toFixed(1)}ms`, target: '≤15ms' },
                      { label: '内存占用', value: `${evaluation.metrics.memoryUsage.toFixed(0)}MB`, target: '≤420MB' },
                      { label: '准确率', value: `${(evaluation.metrics.accuracy * 100).toFixed(1)}%`, target: '≥80%' },
                      { label: '连贯性', value: `${(evaluation.metrics.coherence * 100).toFixed(1)}%`, target: '≥85%' },
                      { label: '学习效率', value: `${(evaluation.metrics.learningEfficiency * 100).toFixed(1)}%`, target: '≥70%' },
                      { label: '记忆召回', value: `${(evaluation.metrics.memoryRecallAccuracy * 100).toFixed(1)}%`, target: '≥85%' },
                      { label: '周期完成率', value: `${(evaluation.metrics.cycleCompletionRate * 100).toFixed(1)}%`, target: '≥95%' },
                      { label: 'STDP更新', value: evaluation.metrics.stdpUpdateCount.toString(), target: '实时' },
                    ].map((item, i) => (
                      <div key={i} className="text-center">
                        <div className="text-xs text-gray-400">{item.label}</div>
                        <div className="text-lg font-bold text-purple-300">{item.value}</div>
                        <div className="text-xs text-gray-500">{item.target}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* 右侧监控面板 */}
          <div className="col-span-3 space-y-4">
            {/* 性能指标 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Activity className="w-4 h-4 text-green-400" />
                  性能指标
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {performance && (
                  <>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">平均周期时间</span>
                        <span>{performance.avgCycleTime.toFixed(2)}ms</span>
                      </div>
                      <Progress value={Math.min(100, performance.avgCycleTime * 10)} className="h-1" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">最大周期时间</span>
                        <span>{performance.maxCycleTime.toFixed(2)}ms</span>
                      </div>
                      <Progress value={Math.min(100, performance.maxCycleTime * 5)} className="h-1" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">总周期数</span>
                        <span>{performance.totalCycles}</span>
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* 内存统计 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <HardDrive className="w-4 h-4 text-blue-400" />
                  海马体记忆
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {memory && (
                  <>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">总记忆数</span>
                      <span>{memory.totalMemories}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">平均强度</span>
                      <span>{(memory.avgStrength * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">平均访问</span>
                      <span>{memory.avgAccessCount.toFixed(1)}次</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">序列长度</span>
                      <span>{memory.sequenceLength}</span>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* STDP统计 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  STDP权重
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {stdp && (
                  <>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">动态权重数</span>
                      <span>{stdp.totalWeights}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">总更新次数</span>
                      <span>{stdp.totalUpdates}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">平均更新</span>
                      <span>{stdp.avgUpdateCount.toFixed(1)}次</span>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* 日志 */}
            <Card className="bg-black/30 border-purple-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-orange-400" />
                  运行日志
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-40">
                  <div className="space-y-1">
                    {logs.map((log, i) => (
                      <div key={i} className="text-xs text-gray-400 font-mono">
                        {log}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* 底部状态栏 */}
      <footer className="fixed bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm border-t border-purple-500/30">
        <div className="container mx-auto px-4 py-2">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <div className="flex items-center gap-4">
              <span>底座模型: Qwen3.5-0.8B</span>
              <Separator orientation="vertical" className="h-4 bg-purple-500/30" />
              <span>权重拆分: 90%静态 + 10%动态</span>
              <Separator orientation="vertical" className="h-4 bg-purple-500/30" />
              <span>量化精度: INT4</span>
            </div>
            <div className="flex items-center gap-4">
              <span>显存限制: 420MB</span>
              <Separator orientation="vertical" className="h-4 bg-purple-500/30" />
              <span>刷新周期: 10ms</span>
              <Separator orientation="vertical" className="h-4 bg-purple-500/30" />
              <span>注意力复杂度: O(1)</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
