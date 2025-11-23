import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Projects from './pages/Projects'
import Agents from './pages/Agents'
import Configs from './pages/Configs'
import KnowledgePlugins from './pages/KnowledgePlugins'
import RunConsole from './pages/RunConsole'
import Layout from './components/Layout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { WorkspaceProvider } from './contexts/WorkspaceContext'
import { ProjectProvider } from './contexts/ProjectContext'

function App() {
  return (
    <Router basename="/app">
      <WorkspaceProvider>
        <ProjectProvider>
          <Layout>
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/projects" element={<Projects />} />
                <Route path="/configs" element={<Configs />} />
                <Route path="/agents" element={<Agents />} />
                <Route path="/knowledge" element={<KnowledgePlugins />} />
                <Route path="/run-console" element={<RunConsole />} />
              </Routes>
            </ErrorBoundary>
          </Layout>
        </ProjectProvider>
      </WorkspaceProvider>
    </Router>
  )
}

export default App

