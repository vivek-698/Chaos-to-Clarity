import React, { useState } from 'react';
import '../App.css';

const Sidebar = () => {
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const [options, setOptions] = useState({
    option1: false,
    option2: false,
    option3: false,
  });

  const handleDropdownToggle = () => {
    setDropdownVisible(!dropdownVisible);
  };

  const handleOptionToggle = (optionName) => {
    setOptions({
      ...options,
      [optionName]: !options[optionName],
    });
  };

  return (
    <div className="sidebar">
      <div className="sidebar-item" onClick={handleDropdownToggle}>
        <span>Dropdown</span>
        <i className={`fa fa-chevron-${dropdownVisible ? 'up' : 'down'}`} />
      </div>
      {dropdownVisible && (
        <div className="sidebar-dropdown">
          <div className="sidebar-dropdown-options">
            <label>
              <input
                type="checkbox"
                checked={options.option1}
                onChange={() => handleOptionToggle('option1')}
              />
              Option 1
            </label>
            <label>
              <input
                type="checkbox"
                checked={options.option2}
                onChange={() => handleOptionToggle('option2')}
              />
              Option 2
            </label>
            <label>
              <input
                type="checkbox"
                checked={options.option3}
                onChange={() => handleOptionToggle('option3')}
              />
              Option 3
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
