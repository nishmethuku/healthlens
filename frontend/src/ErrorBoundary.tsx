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
        <div className="flex min-h-screen items-center justify-center bg-[#060d1a] px-6 text-center text-white">
          <div className="max-w-sm">
            <p className="font-display text-lg font-semibold text-red-300">
              Something went wrong
            </p>
            <p className="mt-2 text-sm text-white/60">
              HealthLens hit an unexpected error. Try reloading the page.
            </p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="btn-gradient font-display mt-6 rounded-xl px-6 py-3 text-sm font-semibold"
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
