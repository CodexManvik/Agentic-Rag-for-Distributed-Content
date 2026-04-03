import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null,
      errorInfo: null,
      errorCount: 0 
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1
    }));
    console.error('Error caught by boundary:', error, errorInfo);
  }

  resetError = () => {
    this.setState({ 
      hasError: false, 
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="dark-theme error-boundary-container">
          <div className="error-boundary-box">
            <h1>⚠️ Something went wrong</h1>
            <p className="error-message">
              The application encountered an unexpected error and needs to recover.
            </p>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="error-details">
                <summary>Error details (dev mode)</summary>
                <pre className="error-stack">
                  <strong>{this.state.error.toString()}</strong>
                  {'\n\n'}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}
            <button 
              onClick={this.resetError}
              className="error-recovery-button"
            >
              Try Again
            </button>
            <p className="error-hint">
              If the error persists, please reload the page or contact support.
            </p>
          </div>
          <style jsx>{`
            .error-boundary-container {
              display: flex;
              align-items: center;
              justify-content: center;
              min-height: 100vh;
              background: #0d1117;
              padding: 20px;
            }
            .error-boundary-box {
              background: #161b22;
              border: 1px solid #30363d;
              border-radius: 8px;
              padding: 40px;
              max-width: 600px;
              box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }
            .error-boundary-box h1 {
              color: #f85149;
              margin: 0 0 16px 0;
              font-size: 28px;
            }
            .error-message {
              color: #8b949e;
              margin: 0 0 20px 0;
              line-height: 1.6;
            }
            .error-details {
              margin: 20px 0;
              background: #0d1117;
              border: 1px solid #30363d;
              border-radius: 6px;
              padding: 12px;
              cursor: pointer;
            }
            .error-details summary {
              color: #58a6ff;
              cursor: pointer;
              padding: 8px;
              user-select: none;
            }
            .error-details summary:hover {
              filter: brightness(1.2);
            }
            .error-stack {
              color: #8b949e;
              font-family: 'Courier New', monospace;
              font-size: 12px;
              overflow-x: auto;
              padding: 12px;
              margin: 8px 0 0 0;
              white-space: pre-wrap;
              word-break: break-word;
            }
            .error-recovery-button {
              background: #238636;
              color: white;
              border: none;
              border-radius: 6px;
              padding: 10px 24px;
              font-size: 14px;
              cursor: pointer;
              font-weight: 600;
              transition: background 200ms;
              margin: 20px 0;
            }
            .error-recovery-button:hover {
              background: #2ea043;
            }
            .error-recovery-button:active {
              background: #238636;
            }
            .error-hint {
              color: #8b949e;
              font-size: 12px;
              margin: 16px 0 0 0;
            }
          `}</style>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
