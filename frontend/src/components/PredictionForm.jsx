import { useState } from "react";
import axios from "axios";

const initialForm = {
  Age: 45,
  Sex: "Male",
  ChestPainType: "NAP",
  RestingBP: 120,
  Cholesterol: 200,
  FastingBS: 0,
  RestingECG: "Normal",
  thalch: 160,
  ExerciseAngina: "N",
  OldPeak: 0.0,
  ST_Slope: "Up",
};

export default function PredictionForm({ onPredict, setLoading }) {
  const [formData, setFormData] = useState(initialForm);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
      );
      onPredict(response.data);
    } catch (error) {
      alert("Error: " + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Age
          </label>
          <input
            type="number"
            name="Age"
            value={formData.Age}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Sex
          </label>
          <select
            name="Sex"
            value={formData.Sex}
            onChange={handleChange}
            className="w-full px-4 py-2 border border border-gray-300 rounded-lg"
          >
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Chest Pain Type
          </label>
          <select
            name="ChestPainType"
            value={formData.ChestPainType}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option value="TA">Typical Angina</option>
            <option value="ATA">Atypical Angina</option>
            <option value="NAP">Non-Anginal Pain</option>
            <option value="ASY">Asymptomatic</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Resting BP
          </label>
          <input
            type="number"
            name="RestingBP"
            value={formData.RestingBP}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Cholesterol
          </label>
          <input
            type="number"
            name="Cholesterol"
            value={formData.Cholesterol}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Fasting BS &gt;120?
          </label>
          <select
            name="FastingBS"
            value={formData.FastingBS}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option value={0}>No</option>
            <option value={1}>Yes</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Resting ECG
          </label>
          <select
            name="RestingECG"
            value={formData.RestingECG}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option>Normal</option>
            <option>ST</option>
            <option>LVH</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Max Heart Rate
          </label>
          <input
            type="number"
            name="thalch"
            value={formData.thalch}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Exercise Angina
          </label>
          <select
            name="ExerciseAngina"
            value={formData.ExerciseAngina}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option value="N">No</option>
            <option value="Y">Yes</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            ST Depression
          </label>
          <input
            type="number"
            step="0.1"
            name="OldPeak"
            value={formData.OldPeak}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            ST Slope
          </label>
          <select
            name="ST_Slope"
            value={formData.ST_Slope}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg"
          >
            <option>Up</option>
            <option>Flat</option>
            <option>Down</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        className="w-full bg-gradient-to-r from-blue-600 to-indigo-700 text-white font-bold py-4 px-8 rounded-xl text-xl hover:from-blue-700 hover:to-indigo-800 transform hover:scale-105 transition duration-200 shadow-lg"
      >
        Assess Cardiovascular Risk
      </button>
    </form>
  );
}
