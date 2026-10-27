import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

// React error boundaries must be class components — there is no hook equivalent.
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("HealthLens crashed:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          className="flex min-h-screen items-center justify-center px-6 text-center text-white"
          style={{ background: "linear-gradient(165deg, #2b86b8 0%, #1c8a98 50%, #153f6d 100%)" }}
        >
          <div className="glass-strong max-w-sm rounded-3xl p-8">
            <p className="font-display text-lg font-semibold text-white">Something went wrong</p>
            <p className="mt-2 text-sm text-white/80">
              HealthLens hit an unexpected error. Try reloading the page.
            </p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="btn-mint font-display mt-6 rounded-full px-8 py-3 text-sm font-semibold"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
