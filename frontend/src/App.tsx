import './App.css'

function App() {
  return (
    <div className="container">
      <header className="header">
        <h1>🏥 Hospital MRI Assistant</h1>
        <p>Medical Tumor Detection from MRI Images</p>
      </header>
      
      <main className="main-content">
        <section className="hero">
          <h2>Advanced AI-Powered MRI Analysis</h2>
          <p>Assist radiologists in identifying tumors from MRI scans using cutting-edge machine learning technology.</p>
        </section>
        
        <section className="features">
          <h2>Features</h2>
          <div className="feature-grid">
            <div className="feature-card">
              <h3>🎯 Accurate Detection</h3>
              <p>High-precision tumor detection using advanced neural networks</p>
            </div>
            <div className="feature-card">
              <h3>⚡ Real-Time Analysis</h3>
              <p>Process MRI scans instantly for rapid clinical decision-making</p>
            </div>
            <div className="feature-card">
              <h3>🔒 Secure & Compliant</h3>
              <p>HIPAA-compliant and secure handling of sensitive medical data</p>
            </div>
            <div className="feature-card">
              <h3>📊 Detailed Reports</h3>
              <p>Comprehensive analysis reports with visual annotations</p>
            </div>
          </div>
        </section>
        
        <section className="cta">
          <h2>Get Started</h2>
          <p>Contact us to learn how Hospital MRI Assistant can improve diagnostic accuracy at your facility.</p>
          <button className="btn-primary">Request Demo</button>
        </section>
      </main>
      
      <footer className="footer">
        <p>&copy; 2024 Hospital MRI Assistant. All rights reserved.</p>
      </footer>
    </div>
  )
}

export default App
