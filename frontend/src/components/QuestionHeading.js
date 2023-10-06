import React from 'react';
import './QuestionHeading.css';
import icon from '../assets/question.png'
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

const QuestionHeading = ({ name, help }) => {
    return (
        <div className='question-header'>
            <h1>{name}</h1>
            <div>
                <OverlayTrigger
                    key="right"
                    placement="right"
                    overlay={
                        <Tooltip id={`tooltip-right`}>
                            {help}
                        </Tooltip>
                    }>
                    <img
                        className='info'
                        src={icon}
                        alt="question mark"
                    />
                </OverlayTrigger>

            </div>
        </div>
    );
};

export default QuestionHeading;
