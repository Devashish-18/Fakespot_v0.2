import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, BarChart3, Zap, Download, ArrowRight, TrendingUp, AlertCircle, CheckCircle, Eye } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: Shield,
      title: 'AI-Powered Detection',
      description: 'Machine learning algorithms analyze multiple account signals simultaneously to determine authenticity with high precision.'
    },
    {
      icon: BarChart3,
      title: 'Interactive Dashboards',
      description: 'Visual metrics including followers, engagement rates, activity patterns, and comprehensive performance graphs.'
    },
    {
      icon: Zap,
      title: 'Intelligent Explanations',
      description: 'Get instant insights explaining WHY an account is real or fake. Every signal is analyzed and explained.'
    },
    {
      icon: Download,
      title: 'Export Reports',
      description: 'Download comprehensive analysis reports in JSON format for verification and documentation.'
    }
  ];

  const signals = [
    {
      icon: TrendingUp,
      title: 'Growth Patterns',
      description: 'Analyzes follower/following ratio and growth velocity'
    },
    {
      icon: Eye,
      title: 'Engagement Metrics',
      description: 'Measures post likes, comments, and interaction rates'
    },
    {
      icon: AlertCircle,
      title: 'Profile Indicators',
      description: 'Checks bio, profile picture, account age, and content'
    },
    {
      icon: CheckCircle,
      title: 'Behavioral Analysis',
      description: 'Detects suspicious patterns and automated behavior'
    }
  ];

  const steps = [
    {
      number: 1,
      title: 'Enter Instagram Username',
      description: 'Type any public Instagram username to analyze',
      icon: '🔍'
    },
    {
      number: 2,
      title: 'AI Analysis In Progress',
      description: 'Our system extracts and analyzes profile signals in real-time',
      icon: '⚙️'
    },
    {
      number: 3,
      title: 'Instant Prediction',
      description: 'Get REAL or FAKE classification with confidence score',
      icon: '✓'
    },
    {
      number: 4,
      title: 'Detailed Performance Data',
      description: 'View interactive graphs showing why the account is classified as real or fake',
      icon: '📊'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-indigo-50 to-white">
      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 sm:py-32">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="fade-in">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-dark mb-6 leading-tight">
              Detect <span className="gradient-primary bg-clip-text text-transparent">Fake Instagram</span> Accounts Instantly
            </h1>
            <p className="text-lg text-gray-600 mb-4">
              Leverage advanced behavioral analysis and AI to identify fraudulent Instagram accounts with remarkable accuracy.
            </p>
            <p className="text-lg text-gray-600 mb-8">
              Get detailed insights into why an account is real or fake through interactive performance graphs and comprehensive analysis.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link to="/analyzer" className="px-8 py-4 gradient-primary text-white rounded-lg font-semibold hover:shadow-lg transform hover:-translate-y-1 transition flex items-center justify-center gap-2">
                Analyze Now <ArrowRight size={20} />
              </Link>
              <a href="#how-it-works" className="px-8 py-4 border-2 border-primary text-primary rounded-lg font-semibold hover:bg-primary hover:text-white transition">
                See How It Works
              </a>
            </div>
            <div className="mt-8 grid grid-cols-3 gap-4 text-sm">
              <div className="p-4 bg-indigo-50 rounded-lg">
                <div className="text-2xl font-bold gradient-primary bg-clip-text text-transparent">98%</div>
                <div className="text-gray-600">Accuracy Rate</div>
              </div>
              <div className="p-4 bg-pink-50 rounded-lg">
                <div className="text-2xl font-bold gradient-primary bg-clip-text text-transparent">Instant</div>
                <div className="text-gray-600">Analysis Speed</div>
              </div>
              <div className="p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold gradient-primary bg-clip-text text-transparent">20+</div>
                <div className="text-gray-600">Data Points</div>
              </div>
            </div>
          </div>
          <div className="fade-in hidden lg:block">
            <div className="relative">
              <div className="gradient-primary rounded-2xl p-1">
                <div className="bg-white rounded-2xl p-8">
                  <div className="space-y-4">
                    <div className="flex items-center gap-3 p-4 bg-green-50 rounded-lg border-2 border-green-200">
                      <CheckCircle className="w-6 h-6 text-success flex-shrink-0" />
                      <div>
                        <div className="font-semibold text-dark">Real Account</div>
                        <div className="text-sm text-gray-600">95% Confidence</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-4 bg-red-50 rounded-lg border-2 border-red-200">
                      <AlertCircle className="w-6 h-6 text-danger flex-shrink-0" />
                      <div>
                        <div className="font-semibold text-dark">Fake Account</div>
                        <div className="text-sm text-gray-600">89% Confidence</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 p-4 bg-gradient-to-br from-indigo-50 to-pink-50 rounded-lg">
                      <div>
                        <div className="text-xs text-gray-600">Followers</div>
                        <div className="font-bold text-dark">40.7M</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-600">Following</div>
                        <div className="font-bold text-dark">10.5M</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-600">Posts</div>
                        <div className="font-bold text-dark">1.2K</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-600">Engagement</div>
                        <div className="font-bold text-dark">4.2%</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Analysis Signals Section */}
      <section className="bg-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-dark mb-4">What We Analyze</h2>
            <p className="text-gray-600 text-lg">Multiple data points work together to identify fake accounts</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {signals.map((signal, idx) => {
              const Icon = signal.icon;
              return (
                <div key={idx} className="p-6 rounded-xl bg-gradient-to-br from-gray-50 to-gray-100 border border-gray-200 hover:shadow-lg hover:border-primary transition fade-in">
                  <Icon className="w-10 h-10 text-primary mb-4" />
                  <h3 className="text-lg font-bold text-dark mb-2">{signal.title}</h3>
                  <p className="text-gray-600 text-sm">{signal.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-dark mb-4">Powerful Features</h2>
            <p className="text-gray-600">Everything you need for comprehensive account analysis</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, idx) => {
              const Icon = feature.icon;
              return (
                <div key={idx} className="p-6 rounded-xl border border-gray-200 hover:shadow-lg hover:border-primary transition fade-in">
                  <Icon className="w-12 h-12 text-primary mb-4" />
                  <h3 className="text-lg font-bold text-dark mb-2">{feature.title}</h3>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works - Extended */}
      <section id="how-it-works" className="bg-gradient-to-b from-white to-indigo-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-dark mb-4">Complete Workflow</h2>
            <p className="text-gray-600 text-lg">From prediction to detailed performance analysis in seconds</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {steps.map((step, idx) => (
              <div key={idx} className="relative fade-in">
                <div className="flex flex-col items-center">
                  <div className="w-20 h-20 gradient-primary rounded-full flex items-center justify-center text-4xl font-bold shadow-lg mb-4">
                    {step.icon}
                  </div>
                  {idx < steps.length - 1 && (
                    <div className="hidden lg:block absolute top-10 left-[55%] w-[45%] h-1 bg-gradient-to-r from-primary to-secondary"></div>
                  )}
                  <div className="text-center">
                    <h3 className="text-lg font-bold text-dark mb-2">Step {step.number}: {step.title}</h3>
                    <p className="text-gray-600 text-sm">{step.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-12 p-8 bg-white rounded-2xl border-2 border-primary shadow-lg">
            <h3 className="text-2xl font-bold text-dark mb-4">🎯 You Get:</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-success flex-shrink-0 mt-1" />
                <div>
                  <div className="font-semibold text-dark">Real/Fake Classification</div>
                  <p className="text-sm text-gray-600">Clear prediction with confidence percentage</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-success flex-shrink-0 mt-1" />
                <div>
                  <div className="font-semibold text-dark">Why Explanation</div>
                  <p className="text-sm text-gray-600">Understand the reasoning behind the prediction</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-success flex-shrink-0 mt-1" />
                <div>
                  <div className="font-semibold text-dark">Performance Graphs</div>
                  <p className="text-sm text-gray-600">4 interactive charts visualizing account metrics</p>
                </div>
              </div>
              <div className="flex gap-4">
                <CheckCircle className="w-6 h-6 text-success flex-shrink-0 mt-1" />
                <div>
                  <div className="font-semibold text-dark">Detailed Metrics</div>
                  <p className="text-sm text-gray-600">8+ profile data points with visual indicators</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-dark mb-4">Powerful Features</h2>
            <p className="text-gray-600">Everything you need to detect fake Instagram accounts</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, idx) => {
              const Icon = feature.icon;
              return (
                <div key={idx} className="p-6 rounded-xl border border-gray-200 hover:shadow-lg hover:border-primary transition fade-in">
                  <Icon className="w-12 h-12 text-primary mb-4" />
                  <h3 className="text-lg font-bold text-dark mb-2">{feature.title}</h3>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-dark mb-4">How It Works</h2>
          <p className="text-gray-600">Three simple steps to detect fake accounts</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, idx) => (
            <div key={idx} className="relative fade-in">
              <div className="flex items-center justify-center mb-6">
                <div className="w-16 h-16 gradient-primary rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                  {step.number}
                </div>
              </div>
              {idx < steps.length - 1 && (
                <div className="hidden md:block absolute top-8 left-[60%] w-[40%] h-1 gradient-primary"></div>
              )}
              <div className="text-center">
                <h3 className="text-xl font-bold text-dark mb-2">{step.title}</h3>
                <p className="text-gray-600">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-indigo-600 to-pink-600 py-16 mt-8">
        <div className="max-w-4xl mx-auto px-4 text-center text-white">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Start Analyzing Now</h2>
          <p className="text-lg mb-8 opacity-90">Get real-time insights into account authenticity with our advanced AI system</p>
          <Link to="/analyzer" className="inline-block px-8 py-4 bg-white text-indigo-600 rounded-lg font-semibold hover:shadow-lg transform hover:-translate-y-1 transition">
            Analyze Your First Account
          </Link>
        </div>
      </section>
    </div>
  );
}
