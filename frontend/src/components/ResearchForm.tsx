/**
 * Research input form component.
 * Dopamine design with playful colors and organic shapes.
 */

import { useState } from 'react';
import { Search, Building2, Package, MapPin, Briefcase, Layers, Target, Sparkles } from 'lucide-react';
import type { ResearchRequest } from '@/types/research';

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
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Row 1: Target Customer + Customer's Industry */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="field-group">
          <label className="label-playful">
            <span className="icon-gradient">
              <Building2 className="w-4 h-4 text-teal-600" />
            </span>
            Target Customer
            <span className="pill-accent ml-auto">Required</span>
          </label>
          <input
            type="text"
            name="account_name"
            value={formData.account_name}
            onChange={handleChange}
            placeholder="e.g., Boeing, Tesla, Remora Carbon"
            required
            className="input-playful"
          />
        </div>

        <div className="field-group">
          <label className="label-playful">
            <span className="icon-gradient">
              <Package className="w-4 h-4 text-pink-500" />
            </span>
            Customer's Industry
            <span className="pill-accent ml-auto">Required</span>
          </label>
          <input
            type="text"
            name="industry"
            value={formData.industry}
            onChange={handleChange}
            placeholder="e.g., aerospace, automotive, carbon capture"
            required
            className="input-playful"
          />
        </div>
      </div>

      {/* Row 2: Customer's Region + Your Company */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="label-playful">
            <span className="icon-gradient">
              <MapPin className="w-4 h-4 text-blue-500" />
            </span>
            Customer's Region
            <span className="text-zinc-400 text-xs font-normal ml-auto">Optional</span>
          </label>
          <input
            type="text"
            name="region"
            value={formData.region}
            onChange={handleChange}
            placeholder="e.g., North America, EMEA, APAC"
            className="input-playful"
          />
        </div>

        <div>
          <label className="label-playful">
            <span className="icon-gradient">
              <Briefcase className="w-4 h-4 text-amber-500" />
            </span>
            Your Company (Seller)
            <span className="pill-accent ml-auto">Required</span>
          </label>
          <input
            type="text"
            name="seller_name"
            value={formData.seller_name}
            onChange={handleChange}
            placeholder="e.g., MathWorks, Salesforce"
            required
            className="input-playful"
          />
        </div>
      </div>

      {/* Row 3: Research Depth */}
      <div>
        <label className="label-playful">
          <span className="icon-gradient">
            <Layers className="w-4 h-4 text-violet-500" />
          </span>
          Research Depth
        </label>
        <select
          name="research_depth"
          value={formData.research_depth}
          onChange={handleChange}
          className="select-playful"
        >
          <option value="quick">Quick - Basic signals</option>
          <option value="standard">Standard - Comprehensive analysis</option>
          <option value="deep">Deep - Exhaustive research</option>
        </select>
      </div>

      {/* Row 4: Context & Research Objective */}
      <div>
        <label className="label-playful">
          <span className="icon-gradient">
            <Target className="w-4 h-4 text-rose-500" />
          </span>
          Context & Research Objective
          <span className="text-zinc-400 text-xs font-normal ml-auto">Optional</span>
        </label>
        <textarea
          name="user_context"
          value={formData.user_context}
          onChange={handleChange}
          rows={3}
          placeholder="Sales meeting notes, specific objectives, relationship context..."
          className="textarea-playful"
        />
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isLoading || !formData.account_name || !formData.industry}
        className="btn-dopamine"
      >
        {isLoading ? (
          <>
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span>Launching Research...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            <span>Start Research</span>
            <span className="text-teal-200 text-sm">→</span>
          </>
        )}
      </button>
    </form>
  );
}
