import { useState } from "react";

function App() {
  const [limitBal, setLimitBal] = useState("");
  const [age, setAge] = useState("");
  const [education, setEducation] = useState("");
  const [marriage, setMarriage] = useState("");
  const [pay0, setPay0] = useState("");

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();

    console.log({
      LIMIT_BAL: Number(limitBal),
      AGE: Number(age),
      EDUCATION: Number(education),
      MARRIAGE: Number(marriage),
      PAY_0: Number(pay0),
    });
  };

  return (
    <div className="app">
      <div className="container">
        <h1>Credit Card Default Prediction</h1>

        <p className="description">
          Enter customer information to predict the probability of credit card
          default.
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="limitBal">Credit Limit</label>
            <input
              id="limitBal"
              type="number"
              value={limitBal}
              onChange={(event) => setLimitBal(event.target.value)}
              placeholder="e.g. 50000"
            />
          </div>

          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input
              id="age"
              type="number"
              value={age}
              onChange={(event) => setAge(event.target.value)}
              placeholder="e.g. 35"
            />
          </div>

          <div className="form-group">
            <label htmlFor="education">Education</label>
            <input
              id="education"
              type="number"
              value={education}
              onChange={(event) => setEducation(event.target.value)}
              placeholder="e.g. 2"
            />
          </div>

          <div className="form-group">
            <label htmlFor="marriage">Marriage</label>
            <input
              id="marriage"
              type="number"
              value={marriage}
              onChange={(event) => setMarriage(event.target.value)}
              placeholder="e.g. 1"
            />
          </div>

          <div className="form-group">
            <label htmlFor="pay0">Payment Status (PAY_0)</label>
            <input
              id="pay0"
              type="number"
              value={pay0}
              onChange={(event) => setPay0(event.target.value)}
              placeholder="e.g. 0"
            />
          </div>

          <button type="submit">Predict</button>
        </form>
      </div>
    </div>
  );
}

export default App;