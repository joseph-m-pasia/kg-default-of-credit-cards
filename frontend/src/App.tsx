
import { useState } from "react";
import "./index.css";

interface PredictionResult {
  prediction: number;
  probability: number;
  risk_category: string;
}

function App() {
  const [limitBal, setLimitBal] = useState("50000");
  const [age, setAge] = useState("35");
  const [pay1, setPay1] = useState("0");
  const [billAmt1, setBillAmt1] = useState("5000");
  const [payAmt1, setPayAmt1] = useState("1000");
  const [education, setEducation] = useState("2");
  const [marriage, setMarriage] = useState("1");
  const [sex, setSex] = useState("1");

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    setResult(null);
    setError("");
    setLoading(true);

    const data = {
      LIMIT_BAL: Number(limitBal),
      AGE: Number(age),
      PAY_1: Number(pay1),
      BILL_AMT1: Number(billAmt1),
      PAY_AMT1: Number(payAmt1),
      EDUCATION: Number(education),
      MARRIAGE: Number(marriage),
      SEX: Number(sex),
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(`API error (${response.status}): ${message}`);
      }

      const predictionResult: PredictionResult = await response.json();

      console.log("Prediction result:", predictionResult);
      setResult(predictionResult);
    } catch (err) {
      console.error("Prediction error:", err);

      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Credit Card Default Prediction</h1>

        <p className="subtitle">
          Enter customer information to predict the probability of credit
          card default.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="limitBal">Credit Limit</label>
            <input
              id="limitBal"
              type="number"
              min="0"
              value={limitBal}
              onChange={(e) => setLimitBal(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input
              id="age"
              type="number"
              min="18"
              max="100"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="pay1">Payment Status</label>
            <select
              id="pay1"
              value={pay1}
              onChange={(e) => setPay1(e.target.value)}
              required
            >
              <option value="-2">-2: No consumption</option>
              <option value="-1">-1: Paid duly</option>
              <option value="0">0: Revolving credit / no delay</option>
              <option value="1">1: Payment delayed by 1 month</option>
              <option value="2">2: Payment delayed by 2 months</option>
              <option value="3">3: Payment delayed by 3 months</option>
              <option value="4">4: Payment delayed by 4 months</option>
              <option value="5">5: Payment delayed by 5 months</option>
              <option value="6">6: Payment delayed by 6 months</option>
              <option value="7">7: Payment delayed by 7 months</option>
              <option value="8">8: Payment delayed by 8+ months</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="billAmt1">Bill Amount</label>
            <input
              id="billAmt1"
              type="number"
              value={billAmt1}
              onChange={(e) => setBillAmt1(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="payAmt1">Payment Amount</label>
            <input
              id="payAmt1"
              type="number"
              min="0"
              value={payAmt1}
              onChange={(e) => setPayAmt1(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="education">Education</label>
            <select
              id="education"
              value={education}
              onChange={(e) => setEducation(e.target.value)}
              required
            >
              <option value="1">Graduate school</option>
              <option value="2">University</option>
              <option value="3">High school</option>
              <option value="4">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="marriage">Marital Status</label>
            <select
              id="marriage"
              value={marriage}
              onChange={(e) => setMarriage(e.target.value)}
              required
            >
              <option value="1">Married</option>
              <option value="2">Single</option>
              <option value="3">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="sex">Sex</label>
            <select
              id="sex"
              value={sex}
              onChange={(e) => setSex(e.target.value)}
              required
            >
              <option value="1">Male</option>
              <option value="2">Female</option>
            </select>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {error && (
          <div className="error" role="alert">
            <h2>Prediction Error</h2>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="result" aria-live="polite">
            <h2>Prediction Result</h2>

            <p>
              <strong>Prediction:</strong>{" "}
              {result.prediction === 1 ? "Default" : "No Default"}
            </p>

            <p>
              <strong>Probability of Default:</strong>{" "}
              {(result.probability * 100).toFixed(1)}%
            </p>

            <p>
              <strong>Risk Category:</strong>{" "}
              {result.risk_category}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
