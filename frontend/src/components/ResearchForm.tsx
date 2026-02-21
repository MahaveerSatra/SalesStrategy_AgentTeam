/**
 * Research input form component.
 */

import { useState } from 'react';
import { Search, Building2, Package, MapPin, FileText, Layers } from 'lucide-react';
import type { ResearchRequest, ResearchDepth } from '@/types/research';

interface ResearchFormProps {
  onSubmit: (request: ResearchRequest) => void;
  isLoading?: boolean;
}

export function ResearchForm({ onSubmit, isLoading = false }: ResearchFormProps) {
  const [formData, setFormData] = useState<ResearchRequest>({
    account_name: '',
    industry: '',
    seller_name: 'MathWorks',
    region: '',
    user_context: '',
    research_depth: 'standard',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Account Name */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <Building2 className="w-4 h-4 text-blue-600" />
          Target Account
        </label>
        <input
          type="text"
          name="account_name"
          value={formData.account_name}
          onChange={handleChange}
          placeholder="e.g., Boeing, Tesla, Remora Carbon"
          required
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        />
      </div>

      {/* Industry */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <Package className="w-4 h-4 text-blue-600" />
          Industry
        </label>
        <input
          type="text"
          name="industry"
          value={formData.industry}
          onChange={handleChange}
          placeholder="e.g., aerospace, automotive, carbon capture"
          required
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        />
      </div>

      {/* Seller Name */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <Search className="w-4 h-4 text-blue-600" />
          Your Company (Seller)
        </label>
        <input
          type="text"
          name="seller_name"
          value={formData.seller_name}
          onChange={handleChange}
          placeholder="e.g., MathWorks, Salesforce"
          required
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        />
      </div>

      {/* Region (Optional) */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <MapPin className="w-4 h-4 text-blue-600" />
          Region <span className="text-slate-400 font-normal">(Optional)</span>
        </label>
        <input
          type="text"
          name="region"
          value={formData.region}
          onChange={handleChange}
          placeholder="e.g., North America, EMEA"
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        />
      </div>

      {/* Research Depth */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <Layers className="w-4 h-4 text-blue-600" />
          Research Depth
        </label>
        <select
          name="research_depth"
          value={formData.research_depth}
          onChange={handleChange}
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
        >
          <option value="quick">Quick (2-3 min) - Basic signals</option>
          <option value="standard">Standard (3-5 min) - Comprehensive</option>
          <option value="deep">Deep (5-10 min) - Exhaustive</option>
        </select>
      </div>

      {/* Sales Context */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-2">
          <FileText className="w-4 h-4 text-blue-600" />
          Sales Context <span className="text-slate-400 font-normal">(Optional)</span>
        </label>
        <textarea
          name="user_context"
          value={formData.user_context}
          onChange={handleChange}
          rows={3}
          placeholder="Add any context: meeting notes, specific objectives, current relationship status..."
          className="w-full px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
        />
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading || !formData.account_name || !formData.industry}
        className="w-full py-4 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-500/25"
      >
        {isLoading ? (
          <>
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Starting Research...
          </>
        ) : (
          <>
            <Search className="w-5 h-5" />
            Start Research
          </>
        )}
      </button>
    </form>
  );
}
